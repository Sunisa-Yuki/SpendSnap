"""Gemini image extraction and validation for receipt data."""

import base64
import json
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from spending import CATEGORIES, CATEGORY_DESCRIPTIONS

# iPhone photos are HEIC; this teaches Pillow to open them.
register_heif_opener()

# Pillow reports iPhone JPEGs (which embed depth/HDR data) as "MPO".
SUPPORTED_FORMATS = {"JPEG", "MPO", "PNG", "HEIF"}


class ReceiptAIError(Exception):
    """An extraction or response-validation error safe to show in the app."""


def _response_text(response: Any, *, is_gemma: bool) -> str:
    """Get user-facing model text, excluding Gemma's internal thought parts."""
    if is_gemma:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            visible_parts = [
                part.text
                for part in parts
                if getattr(part, "text", None)
                and not getattr(part, "thought", False)
            ]
            if visible_parts:
                return "\n".join(visible_parts).strip()
    return (getattr(response, "output_text", None) or getattr(response, "text", None) or "").strip()


def prepare_receipt_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Orient and downscale large receipt photos before sending them to Google."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in SUPPORTED_FORMATS:
                raise ReceiptAIError("Please upload a JPG, PNG, or HEIC receipt image.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
    except (OSError, ValueError) as exc:
        raise ReceiptAIError(
            "The uploaded image could not be prepared. Please try another JPG, PNG, or HEIC."
        ) from exc
    return output.getvalue(), "image/jpeg"


def _api_error_message(status_code: int) -> str:
    if status_code in {400, 422}:
        return (
            "Google rejected the request. Check that the uploaded file is a "
            "valid JPG or PNG and that the selected model supports image input."
        )
    if status_code in {401, 403}:
        return (
            "Google rejected the API key or this key does not have Gemini API "
            "access. Create a fresh key in Google AI Studio, replace the value "
            "in .env, and restart Streamlit."
        )
    if status_code == 404:
        return (
            "Google says the configured model is unavailable to this account or "
            "API. Check GEMINI_MODEL in .env and confirm model access in Google AI Studio."
        )
    if status_code == 429:
        return (
            "The Gemini API quota or rate limit was reached. Check API usage "
            "and billing in Google AI Studio, then try again later."
        )
    if status_code >= 500:
        return "The Gemini service is temporarily unavailable. Please try again later."
    return (
        f"The Gemini API returned HTTP {status_code}. Check your API key, "
        "model access, and request configuration."
    )


def _number(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ReceiptAIError(f"The receipt response contained an invalid {field_name}.")
    try:
        cleaned = str(value).strip().replace(",", "").replace("$", "")
        number = float(Decimal(cleaned))
    except (InvalidOperation, ValueError, TypeError):
        raise ReceiptAIError(
            f"The receipt response contained an invalid {field_name}."
        ) from None
    if not math.isfinite(number):
        raise ReceiptAIError(f"The receipt response contained an invalid {field_name}.")
    return number


def validate_receipt(payload: Any) -> dict[str, Any]:
    """Validate model output and normalize it without filling in missing facts."""
    if not isinstance(payload, dict):
        raise ReceiptAIError("The AI response was not a JSON object.")

    merchant = payload.get("merchant")
    transaction_date = payload.get("date")
    currency = payload.get("currency")
    for value, label in (
        (merchant, "merchant"),
        (transaction_date, "date"),
        (currency, "currency"),
    ):
        if value is not None and not isinstance(value, str):
            raise ReceiptAIError(f"The receipt response contained an invalid {label}.")

    transaction_date = transaction_date.strip() if transaction_date else None
    if transaction_date:
        try:
            date.fromisoformat(transaction_date)
        except ValueError:
            raise ReceiptAIError(
                "The date could not be read reliably. You can correct it before saving."
            ) from None

    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ReceiptAIError("The receipt response had an invalid items list.")

    items = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ReceiptAIError("The receipt response contained an invalid line item.")
        name = raw_item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ReceiptAIError(
                "A line item had no readable name. Try a clearer image or edit the receipt."
            )
        category = raw_item.get("category")
        if isinstance(category, str):
            category = next(
                (known for known in CATEGORIES if known.casefold() == category.strip().casefold()),
                None,
            )
        else:
            category = None
        quantity = _number(raw_item.get("quantity"), "quantity")
        if quantity is not None and quantity <= 0:
            raise ReceiptAIError("A line item had an invalid quantity.")
        category = raw_item.get("category")
        items.append(
            {
                "name": name.strip(),
                "quantity": quantity,
                "unit_price": _number(raw_item.get("unit_price"), "unit price"),
                "line_total": _number(raw_item.get("line_total"), "line total"),
                "category": category if category in CATEGORIES else None,
            }
        )

    return {
        "merchant": merchant.strip() if merchant and merchant.strip() else None,
        "date": transaction_date,
        "currency": (
            currency.strip().upper()
            if currency and re.fullmatch(r"[A-Za-z]{3}", currency.strip())
            else None
        ),
        "total": _number(payload.get("total"), "receipt total"),
        "tax": _number(payload.get("tax"), "tax"),
        "tip": _number(payload.get("tip"), "tip"),
        "items": items,
    }


def extract_receipt(
    image_bytes: bytes, mime_type: str, model_name: str, api_key: str
) -> dict[str, Any]:
    """Send one image to the configured Gemini API model and validate its JSON."""
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ReceiptAIError("Please upload a JPG, JPEG, or PNG receipt image.")

    category_list = ", ".join(f'"{name}"' for name in CATEGORIES)
    category_guide = "\n".join(
        f"- {name}: {description}" for name, description in CATEGORY_DESCRIPTIONS.items()
    )
    prompt = f"""
Read this receipt image and return only one JSON object with this shape:
{{
  "merchant": "string or null",
  "date": "YYYY-MM-DD or null",
  "currency": "ISO 4217 code or null",
  "total": number or null,
  "tax": number or null,
  "tip": number or null,
  "items": [
    {{
      "name": "string",
      "quantity": number or null,
      "unit_price": number or null,
      "line_total": number or null,
      "category": one of [{category_list}]
    }}
  ]
}}
Choose each item's category from this list:
{category_guide}
Receipt item names are often abbreviated (for example "CHKN" is chicken), so
work out what the product is before choosing. Use the type of store as
context: ready-to-eat food from a grocery or warehouse store such as Costco,
Walmart, or Trader Joe's is "Groceries", while the same food from a restaurant
is "Dining out". Use "Other" only when no category fits.
For every other field, use null when a value is not shown or cannot be read.
Do not guess, calculate, or substitute missing values. Include only line items visible on the receipt.
If the image is not a receipt or is unreadable, return an empty items list and
null for fields you cannot identify. Do not include markdown fences.
""".strip()

    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError

        client = genai.Client(api_key=api_key)
        is_gemma = model_name.casefold().startswith("gemma-")
        if is_gemma:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                    # Long receipts can exhaust the output limit on hidden
                    # reasoning before any JSON is written.
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
        else:
            response = client.interactions.create(
                model=model_name,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                        "mime_type": mime_type,
                    },
                ],
            )
    except APIError as exc:
        raise ReceiptAIError(_api_error_message(exc.code)) from None
    except Exception:
        raise ReceiptAIError(
            "Could not connect to the Gemini API. Check your internet connection "
            "and try again."
        ) from None

    response_text = _response_text(response, is_gemma=is_gemma)
    if not response_text:
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        finish_reason = getattr(finish_reason, "name", str(finish_reason))
        if finish_reason == "MAX_TOKENS":
            raise ReceiptAIError(
                "Gemma used its response limit before returning receipt data. "
                "Try analyzing it again, or for a very long receipt, upload it "
                "as two photos and save each part."
            )
        raise ReceiptAIError(
            "The model returned no receipt text. Try a clearer image or analyze "
            "it again; you can also change GEMINI_MODEL in .env to gemini-3.8-flash."
        )
    cleaned = response_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ReceiptAIError(
            "The AI response was not valid JSON. Try analyzing the image again."
        ) from None
    return validate_receipt(payload)
