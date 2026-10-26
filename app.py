import os
import uuid
from datetime import date
from math import isfinite
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from receipt_ai import ReceiptAIError, extract_receipt, prepare_receipt_image
from storage import StorageError, load_receipts, save_receipts
from spending import (
    CATEGORIES,
    PERIODS,
    amount_for_item,
    categorize_item,
    current_category,
    line_item_sum,
    receipt_date,
    receipts_in_period,
)

# A chart can only use 8 colors that stay distinct for color-blind readers, so
# the donut shows the 7 most common student categories and folds the rest into
# "Everything else". The table below the chart still lists every category.
CHART_COLORS = {
    "Groceries": "#2a78d6",
    "Dining out": "#eb6834",
    "Coffee & drinks": "#1baf7a",
    "Household & paper products": "#eda100",
    "Personal care & health": "#e87ba4",
    "School & office supplies": "#008300",
    "Transportation": "#e34948",
    "Everything else": "#6250d6",
}


def amount_for_saved_item(item: dict[str, Any]) -> float | None:
    return amount_for_item(item)


def display_number(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def parse_manual_number(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        number = float(value.strip().replace(",", "").replace("$", ""))
    except ValueError:
        return None
    if not isfinite(number):
        return None
    return number


def is_valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def edited_items_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in frame.iterrows():
        name = str(row.get("Item", "")).strip()
        if not name or name.casefold() == "nan":
            continue

        def optional_float(field: str) -> float | None:
            value = row.get(field)
            if pd.isna(value) or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        category = row.get("Category")
        if category not in CATEGORIES:
            category = categorize_item(name)
        quantity = optional_float("Quantity")
        unit_price = optional_float("Unit price")
        line_total = optional_float("Line total")
        if unit_price is None and quantity is not None and quantity > 0 and line_total is not None:
            unit_price = round(line_total / quantity, 2)
        records.append(
            {
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total,
                "category": category,
            }
        )
    return records


load_dotenv()
MODEL_NAME = os.getenv("GEMINI_MODEL") or "gemma-4-26b-a4b-it"

st.set_page_config(
    page_title="SpendSnap Student Edition",
    page_icon="💸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background: #f7f8f5;
        color: #24352d;
    }
    [data-testid="stHeader"] { background: #f7f8f5; }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] label,
    [data-testid="stWidgetLabel"] p {
        color: #34463d;
    }
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button p {
        color: #34463d !important;
        opacity: 1 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] p {
        color: #176b4b !important;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #dce5dd;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e8ece6; border-radius: 14px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #24352d;
    }
    .hero { padding: 10px 0 4px; }
    .hero h1 { letter-spacing: -0.04em; color: #14382c; }
    .hero p { color: #52665b; }
    .privacy-note {
        background: #fff9e9; border-left: 4px solid #e6b84d;
        border-radius: 6px; padding: 12px 14px; color: #51441f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "receipts" not in st.session_state:
    try:
        st.session_state.receipts = load_receipts()
        for saved_receipt in st.session_state.receipts:
            for saved_item in saved_receipt["items"]:
                saved_item["category"] = current_category(
                    saved_item.get("category"), saved_item["name"]
                )
    except StorageError as exc:
        # Stop instead of starting empty, so the next save can't overwrite the file.
        st.error(f"{exc} Fix or move that file, then refresh the page.")
        st.stop()
if "pending_receipt" not in st.session_state:
    st.session_state.pending_receipt = None

st.markdown(
    '<div class="hero"><h1>SpendSnap</h1>'
    "<p>Your everyday spending, made easier to understand.</p></div>",
    unsafe_allow_html=True,
)

dashboard_tab, scan_tab = st.tabs(["Your spending", "Scan a receipt"])

with dashboard_tab:
    all_receipts = st.session_state.receipts
    period = (
        st.segmented_control("Show", PERIODS, default="All time", key="dashboard_period")
        or "All time"
    )
    receipts = receipts_in_period(all_receipts, period, date.today())
    undated_count = sum(1 for receipt in all_receipts if receipt_date(receipt) is None)
    if period != "All time" and undated_count:
        st.caption(
            f"{undated_count} receipt(s) without a readable date only appear in All time."
        )
    known_receipts = [
        receipt
        for receipt in receipts
        if receipt.get("total") is not None
    ]
    totals_by_currency: dict[str, float] = {}
    for receipt in known_receipts:
        code = receipt.get("currency") or "Currency unknown"
        totals_by_currency[code] = totals_by_currency.get(code, 0.0) + float(
            receipt["total"]
        )
    one_currency = len(totals_by_currency) == 1
    currency_label = next(iter(totals_by_currency), "")
    total_spend = next(iter(totals_by_currency.values()), 0.0)

    metric_total, metric_count, metric_average = st.columns(3)
    metric_total.metric(
        "Known receipt totals",
        f"{currency_label} {total_spend:,.2f}"
        if one_currency
        else "Mixed currencies" if totals_by_currency else "—",
    )
    metric_count.metric("Receipts analyzed", len(receipts))
    metric_average.metric(
        "Average receipt",
        f"{currency_label} {total_spend / len(known_receipts):,.2f}"
        if one_currency and known_receipts
        else "—",
    )
    if len(totals_by_currency) > 1:
        st.caption(
            "Known totals by currency: "
            + " · ".join(
                f"{code} {amount:,.2f}"
                for code, amount in sorted(totals_by_currency.items())
            )
        )

    if all_receipts and not receipts:
        st.info(f"No receipts dated {period.lower()} yet.")
    elif not receipts:
        st.info(
            "Your dashboard is ready. Scan a receipt to see your spending "
            "totals, categories, and a student-friendly estimate."
        )
    else:
        category_rows = []
        item_rows = []
        for receipt_index, receipt in enumerate(receipts):
            for item in receipt["items"]:
                amount = amount_for_saved_item(item)
                category_rows.append(
                    {
                        "Category": item["category"],
                        "Amount": amount,
                        "Currency": receipt.get("currency") or "Unknown currency",
                        "Receipt": receipt_index,
                    }
                )
                item_rows.append(
                    {
                        "Date": receipt.get("date") or "Not shown",
                        "Merchant": receipt.get("merchant") or "Not identified",
                        "Item": item["name"],
                        "Quantity": item.get("quantity"),
                        "Unit price": item.get("unit_price"),
                        "Line total": item.get("line_total"),
                        "Category": item["category"],
                        "Amount": amount,
                        "Currency": receipt.get("currency") or "Unknown currency",
                        "Receipt total": receipt.get("total"),
                        "Tax": receipt.get("tax"),
                        "Tip": receipt.get("tip"),
                    }
                )

        st.subheader("Where it goes")
        category_frame = pd.DataFrame(category_rows)
        summary = category_frame.groupby(["Category", "Currency"], as_index=False).agg(
            Purchases=("Category", "size"),
            Receipts=("Receipt", "nunique"),
            # Blank (not 0.00) when no item in the category has a known price.
            Spent=("Amount", lambda amounts: amounts.sum(min_count=1)),
        )
        summary["Share"] = summary["Spent"] / summary.groupby("Currency")[
            "Spent"
        ].transform("sum")
        summary = summary.sort_values("Spent", ascending=False)
        if summary["Spent"].sum() > 0:
            chart_data = (
                summary[summary["Spent"] > 0]
                .assign(
                    Slice=lambda frame: frame["Category"].where(
                        frame["Category"].isin(CHART_COLORS), "Everything else"
                    )
                )
                .groupby(["Slice", "Currency"], as_index=False)[["Spent", "Share"]]
                .sum()
            )
            chart = alt.Chart(chart_data).mark_arc(
                innerRadius=70, stroke="#f7f8f5", strokeWidth=2
            ).encode(
                theta=alt.Theta("Spent:Q", stack=True),
                color=alt.Color(
                    "Slice:N",
                    # Fixed colors per category, so filtering never repaints them.
                    scale=alt.Scale(
                        domain=list(CHART_COLORS), range=list(CHART_COLORS.values())
                    ),
                    legend=alt.Legend(title=None),
                ),
                order=alt.Order("Spent:Q", sort="descending"),
                tooltip=[
                    alt.Tooltip("Slice:N", title="Category"),
                    "Currency",
                    alt.Tooltip("Spent:Q", format=".2f"),
                    alt.Tooltip("Share:Q", format=".0%"),
                ],
            ).properties(height=280)
            if chart_data["Currency"].nunique() > 1:
                chart = chart.facet(
                    column=alt.Column("Currency:N", title=None),
                    data=chart_data,
                )
            st.altair_chart(chart, width="stretch")
        else:
            st.caption(
                "No line-item prices are available yet. Edit item prices while "
                "reviewing a receipt to include them in category totals."
            )
        st.dataframe(
            summary if summary["Currency"].nunique() > 1 else summary.drop(columns="Currency"),
            width="stretch",
            hide_index=True,
            column_config={
                "Purchases": st.column_config.NumberColumn(
                    "Purchases", help="Line items in this category"
                ),
                "Receipts": st.column_config.NumberColumn(
                    "Receipts", help="Separate receipts (trips) with this category"
                ),
                "Spent": st.column_config.NumberColumn("Spent", format="%.2f"),
                "Share": st.column_config.NumberColumn("Share", format="percent"),
            },
        )

        st.subheader("Receipt details")
        details = pd.DataFrame(
            [
                {
                    "Date": receipt.get("date") or "Not shown",
                    "Merchant": receipt.get("merchant") or "Not identified",
                    "Total": receipt.get("total"),
                    "Currency": receipt.get("currency") or "Not identified",
                    "Items": len(receipt["items"]),
                }
                for receipt in receipts
            ]
        )
        st.dataframe(details, width="stretch", hide_index=True)
        if item_rows:
            st.caption("All extracted line items (blank amounts were not identified)")
            st.dataframe(pd.DataFrame(item_rows), width="stretch", hide_index=True)
            csv_data = pd.DataFrame(item_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download line items as CSV",
                data=csv_data,
                file_name="spendsnap-line-items.csv",
                mime="text/csv",
            )

        st.subheader("A small-spend what-if")
        all_items = [
            (receipt, item)
            for receipt in receipts
            for item in receipt["items"]
            if amount_for_saved_item(item) is not None
        ]
        if all_items:
            choices = [
                f"{receipt.get('merchant') or 'Receipt'} — {item['name']} "
                f"({receipt.get('currency') or ''} "
                f"{amount_for_saved_item(item):.2f})"
                for receipt, item in all_items
            ]
            selected_index = st.selectbox(
                "Pick a purchase to estimate",
                options=range(len(choices)),
                format_func=lambda index: choices[index],
            )
            repeats_per_week = st.number_input(
                "If you bought it this many times each week…",
                min_value=1,
                max_value=21,
                value=5,
                step=1,
            )
            selected_receipt, selected_item = all_items[selected_index]
            monthly_estimate = (
                amount_for_saved_item(selected_item) * repeats_per_week * 4.33
            )
            selected_currency = selected_receipt.get("currency") or ""
            st.success(
                f"That would be about {selected_currency} {monthly_estimate:,.2f} "
                "per month."
            )
            st.caption(
                "Hypothetical estimate only: it assumes the selected receipt's "
                "line-item amount and your chosen weekly frequency for 4.33 weeks. "
                "It is not a claim about your purchase history or guaranteed savings."
            )
        else:
            st.caption(
                "Once a receipt has a line-item price, try a weekly what-if estimate here."
            )

with scan_tab:
    st.subheader("Add a receipt")
    st.markdown(
        '<div class="privacy-note">Privacy note: when you analyze a receipt, '
        "its image is sent to the configured Google Gemini API model for "
        "processing. Only upload a receipt you are comfortable sharing with "
        "that provider.</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Configured model: `{MODEL_NAME}`")
    st.write("")
    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=["jpg", "jpeg", "png", "heic", "heif"],
        help="JPG, JPEG, PNG, and HEIC (iPhone) images are supported.",
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        try:
            # Browsers can't display HEIC, so preview the converted JPEG.
            preview_bytes, _ = prepare_receipt_image(image_bytes)
            st.image(preview_bytes, caption="Receipt preview", width="stretch")
        except ReceiptAIError:
            st.error("That file could not be opened as an image. Please try another one.")
            image_bytes = b""

        if image_bytes:
            st.caption(
                f"Selected image: {len(image_bytes) / (1024 * 1024):.1f} MB. "
                "Large photos can take longer to upload and analyze."
            )

        analyze_clicked = st.button(
            "Analyze receipt",
            type="primary",
            disabled=not image_bytes,
            width="stretch",
        )
        if analyze_clicked:
            st.session_state.pending_receipt = None
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error(
                    "Add GEMINI_API_KEY to your local .env file, then restart "
                    "Streamlit to analyze a receipt."
                )
            else:
                try:
                    prepared_bytes, mime_type = prepare_receipt_image(image_bytes)
                    st.caption(
                        f"Sending an optimized {len(prepared_bytes) / 1024:.0f} KB "
                        "image to the AI provider."
                    )
                    with st.spinner(
                        f"Sending the receipt to {MODEL_NAME} and reading its items…"
                    ):
                        extracted = extract_receipt(
                            image_bytes=prepared_bytes,
                            mime_type=mime_type,
                            model_name=MODEL_NAME,
                            api_key=api_key,
                        )
                    extracted["currency"] = extracted.get("currency")
                    st.session_state.pending_receipt = extracted
                    st.session_state.pending_id = uuid.uuid4().hex
                    st.success("Receipt analyzed. Review the details before saving.")
                except ReceiptAIError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error(
                        "The receipt could not be analyzed due to an unexpected "
                        "application error. Please try again."
                    )

    pending = st.session_state.pending_receipt
    if pending:
        st.divider()
        st.subheader("Review before saving")
        st.caption(
            "AI extraction can make mistakes. Check the receipt and correct "
            "anything that looks off. Blank fields mean the value was not identified."
        )
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            merchant = st.text_input(
                "Merchant",
                value=pending.get("merchant") or "",
                key=f"merchant-{st.session_state.pending_id}",
            )
            transaction_date = st.text_input(
                "Date (YYYY-MM-DD, if known)",
                value=pending.get("date") or "",
                key=f"date-{st.session_state.pending_id}",
            )
            currency = st.text_input(
                "Currency code (if known)",
                value=pending.get("currency") or "",
                key=f"currency-{st.session_state.pending_id}",
            )
        with info_col2:
            total_text = st.text_input(
                "Reported receipt total (leave blank if not shown)",
                value=display_number(pending.get("total")),
                key=f"total-{st.session_state.pending_id}",
            )
            tax_text = st.text_input(
                "Tax (if shown)",
                value=display_number(pending.get("tax")),
                key=f"tax-{st.session_state.pending_id}",
            )
            tip_text = st.text_input(
                "Tip (if shown)",
                value=display_number(pending.get("tip")),
                key=f"tip-{st.session_state.pending_id}",
            )

        initial_items = []
        for item in pending["items"]:
            initial_items.append(
                {
                    "Item": item["name"],
                    "Quantity": item.get("quantity"),
                    "Unit price": item.get("unit_price"),
                    "Line total": item.get("line_total"),
                    "Category": item.get("category") or categorize_item(item["name"]),
                }
            )
        edited_items = st.data_editor(
            pd.DataFrame(
                initial_items,
                columns=["Item", "Quantity", "Unit price", "Line total", "Category"],
            ),
            key=f"items-{st.session_state.pending_id}",
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Item": st.column_config.TextColumn("Item"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.01),
                "Unit price": st.column_config.NumberColumn("Unit price", min_value=0.0),
                "Line total": st.column_config.NumberColumn("Line total", min_value=0.0),
                "Category": st.column_config.SelectboxColumn(
                    "Category", options=CATEGORIES, required=True
                ),
            },
        )

        reviewed_items = edited_items_to_records(edited_items)
        reported_total = parse_manual_number(total_text)
        if total_text.strip() and reported_total is None:
            st.warning("The receipt total must be a number, or left blank if unknown.")
        else:
            visible_sum = line_item_sum(reviewed_items)
            if reported_total is not None and visible_sum is not None:
                check_tax = parse_manual_number(tax_text) or 0
                check_tip = parse_manual_number(tip_text) or 0
                expected_total = visible_sum + check_tax + check_tip
                if round(abs(reported_total - expected_total), 2) > 0.01:
                    st.warning(
                        f"Items {visible_sum:.2f} + tax {check_tax:.2f} + tip "
                        f"{check_tip:.2f} = {expected_total:.2f}, but the receipt "
                        f"says {reported_total:.2f}. "
                        "Check the items and total; SpendSnap will not change either."
                    )
            elif reported_total is not None and reviewed_items:
                st.info(
                    "Some item prices are missing, so SpendSnap cannot compare "
                    "the complete line-item sum with the receipt total."
                )

        if st.button(
            "Save transaction",
            type="primary",
            disabled=not merchant.strip() or (bool(total_text.strip()) and reported_total is None),
        ):
            manual_tax = parse_manual_number(tax_text)
            manual_tip = parse_manual_number(tip_text)
            if (tax_text.strip() and manual_tax is None) or (tip_text.strip() and manual_tip is None):
                st.error("Tax and tip must be numbers, or left blank if unknown.")
            elif transaction_date.strip() and not is_valid_date(transaction_date.strip()):
                st.error("Enter the date as YYYY-MM-DD, or leave it blank if unknown.")
            elif not reviewed_items:
                st.error("Add at least one named line item before saving.")
            else:
                new_receipt = {
                    "merchant": merchant.strip(),
                    "date": transaction_date.strip() or None,
                    "currency": currency.strip().upper() or None,
                    "total": reported_total,
                    "tax": manual_tax,
                    "tip": manual_tip,
                    "items": reviewed_items,
                }
                try:
                    save_receipts(st.session_state.receipts + [new_receipt])
                except OSError:
                    st.error("Could not save the receipt to disk. Please try again.")
                else:
                    st.session_state.receipts.append(new_receipt)
                    st.session_state.pending_receipt = None
                    st.success("Saved to your dashboard.")
                    st.rerun()
