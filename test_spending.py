import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from google.genai.errors import APIError

from receipt_ai import (
    ReceiptAIError,
    extract_receipt,
    prepare_receipt_image,
    validate_receipt,
)
from spending import (
    amount_for_item,
    categorize_item,
    current_category,
    is_small_purchase,
    line_item_sum,
    monthly_estimate,
    receipts_in_period,
    saving_suggestion,
)
from storage import StorageError, load_receipts, save_receipts


class ReceiptValidationTests(unittest.TestCase):
    def test_prepare_image_orients_and_converts_png(self):
        from io import BytesIO

        from PIL import Image

        source = BytesIO()
        Image.new("RGB", (2400, 1200), "white").save(source, format="PNG")
        optimized, mime_type = prepare_receipt_image(source.getvalue())
        self.assertEqual(mime_type, "image/jpeg")
        with Image.open(BytesIO(optimized)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(max(image.size), 2200)

    def test_valid_receipt_keeps_missing_values_unknown(self):
        result = validate_receipt(
            {
                "merchant": "Campus Cafe",
                "date": "2026-10-25",
                "currency": "usd",
                "total": "12.50",
                "items": [
                    {
                        "name": "Iced matcha",
                        "quantity": 1,
                        "unit_price": 6.25,
                        "line_total": 6.25,
                    }
                ],
            }
        )
        self.assertEqual(result["currency"], "USD")
        self.assertIsNone(result["tax"])
        self.assertEqual(result["items"][0]["name"], "Iced matcha")

    def test_model_category_is_kept_only_when_known(self):
        result = validate_receipt(
            {
                "items": [
                    {"name": "Bananas", "category": "groceries"},
                    {"name": "Gift card", "category": "Gifts"},
                    {"name": "Napkins"},
                ]
            }
        )
        categories = [item["category"] for item in result["items"]]
        self.assertEqual(categories, ["Groceries", None, None])

    def test_ambiguous_currency_symbol_stays_unknown(self):
        result = validate_receipt({"currency": "$", "items": []})
        self.assertIsNone(result["currency"])

    def test_rejects_invalid_total_and_date(self):
        with self.assertRaises(ReceiptAIError):
            validate_receipt({"total": "not-a-number", "items": []})
        with self.assertRaises(ReceiptAIError):
            validate_receipt({"date": "sometime", "items": []})

    @patch("google.genai.Client")
    def test_image_extraction_parses_model_json(self, client_factory):
        client = client_factory.return_value
        client.interactions.create.return_value = SimpleNamespace(
            output_text='{"merchant":"Cafe","items":[{"name":"Matcha"}]}'
        )
        result = extract_receipt(b"image-bytes", "image/jpeg", "model-name", "test-key")
        self.assertEqual(result["merchant"], "Cafe")
        self.assertEqual(result["items"][0]["name"], "Matcha")
        client.interactions.create.assert_called_once()
        request = client.interactions.create.call_args.kwargs
        self.assertEqual(request["model"], "model-name")
        self.assertEqual(request["input"][1]["type"], "image")
        self.assertEqual(request["input"][1]["mime_type"], "image/jpeg")
        self.assertEqual(request["input"][1]["data"], "aW1hZ2UtYnl0ZXM=")

    @patch("google.genai.Client")
    def test_gemma_model_uses_generate_content_with_image(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="internal reasoning", thought=True),
                            SimpleNamespace(
                                text='{"merchant":"Cafe","items":[{"name":"Matcha"}]}',
                                thought=None,
                            ),
                        ]
                    ),
                    finish_reason="STOP",
                )
            ],
        )
        result = extract_receipt(
            b"image-bytes",
            "image/jpeg",
            "gemma-4-26b-a4b-it",
            "test-key",
        )
        self.assertEqual(result["merchant"], "Cafe")
        client.models.generate_content.assert_called_once()
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.thinking_config.thinking_level, "MINIMAL")
        client.interactions.create.assert_not_called()

    @patch("google.genai.Client")
    def test_gemma_empty_response_suggests_retry(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[]),
                    finish_reason="STOP",
                )
            ],
        )
        with self.assertRaisesRegex(ReceiptAIError, "returned no receipt text"):
            extract_receipt(
                b"image-bytes", "image/jpeg", "gemma-4-26b-a4b-it", "test-key"
            )

    @patch("google.genai.Client")
    def test_gemma_token_limit_has_clear_message(self, client_factory):
        client = client_factory.return_value
        client.models.generate_content.return_value = SimpleNamespace(
            text=None,
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[]),
                    finish_reason="MAX_TOKENS",
                )
            ],
        )
        with self.assertRaisesRegex(ReceiptAIError, "response limit"):
            extract_receipt(
                b"image-bytes", "image/jpeg", "gemma-4-26b-a4b-it", "test-key"
            )

    @patch("google.genai.Client")
    def test_image_extraction_reports_invalid_json(self, client_factory):
        client = client_factory.return_value
        client.interactions.create.return_value = SimpleNamespace(output_text="not JSON")
        with self.assertRaisesRegex(ReceiptAIError, "not valid JSON"):
            extract_receipt(b"image-bytes", "image/jpeg", "model-name", "test-key")

    @patch("google.genai.Client")
    def test_image_extraction_explains_invalid_api_key(self, client_factory):
        client = client_factory.return_value
        client.interactions.create.side_effect = APIError(
            403,
            {"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}},
        )
        with self.assertRaisesRegex(ReceiptAIError, "Google rejected the API key"):
            extract_receipt(b"image-bytes", "image/jpeg", "model-name", "test-key")

    @patch("google.genai.Client")
    def test_image_extraction_explains_missing_model(self, client_factory):
        client = client_factory.return_value
        client.interactions.create.side_effect = APIError(
            404,
            {"error": {"code": 404, "message": "Missing", "status": "NOT_FOUND"}},
        )
        with self.assertRaisesRegex(ReceiptAIError, "model is unavailable"):
            extract_receipt(b"image-bytes", "image/jpeg", "model-name", "test-key")


class SpendingTests(unittest.TestCase):
    def test_categories_cover_common_student_purchases(self):
        self.assertEqual(categorize_item("Iced matcha"), "Coffee & drinks")
        self.assertEqual(categorize_item("Tomatoes"), "Groceries")
        self.assertEqual(categorize_item("E ROTISSERIE CHKN 287461"), "Groceries")
        self.assertEqual(
            categorize_item("E PAPER TOWEL 12PK 583920"), "Household & paper products"
        )
        self.assertEqual(
            categorize_item("Toilet Paper 30 Rolls"), "Household & paper products"
        )
        self.assertEqual(categorize_item("Mystery item"), "Other")
        self.assertEqual(categorize_item("Teriyaki bowl"), "Dining out")

    def test_categories_ignore_keywords_inside_other_words(self):
        self.assertEqual(categorize_item("Ribeye steak"), "Dining out")
        self.assertEqual(categorize_item("Hibiscus"), "Other")

    def test_ai_category_is_kept_only_when_valid(self):
        result = validate_receipt(
            {
                "items": [
                    {"name": "Pad thai", "category": "Dining out"},
                    {"name": "Thing", "category": "Snacks"},
                ]
            }
        )
        self.assertEqual(result["items"][0]["category"], "Dining out")
        self.assertIsNone(result["items"][1]["category"])

    def test_old_saved_categories_map_to_current_list(self):
        self.assertEqual(current_category("Drinks", "Iced matcha"), "Coffee & drinks")
        self.assertEqual(current_category("Food & restaurants", "Pad thai"), "Dining out")
        self.assertEqual(current_category("Shopping", "Phone charger"), "Electronics")
        self.assertEqual(current_category("Groceries", "Eggs"), "Groceries")

    def test_line_sum_only_uses_complete_known_amounts(self):
        self.assertEqual(amount_for_item({"unit_price": 3, "quantity": 2}), 6)
        self.assertEqual(
            line_item_sum(
                [
                    {"line_total": 4.5},
                    {"unit_price": 2, "quantity": 2},
                ]
            ),
            8.5,
        )
        self.assertIsNone(line_item_sum([{"name": "Unknown price"}]))

    def test_period_filter_uses_calendar_week_and_month(self):
        receipts = [
            {"date": "2026-10-19"},  # Monday of the current week
            {"date": "2026-10-18"},  # Sunday before: this month, last week
            {"date": "2026-09-30"},
            {"date": None},
        ]
        today = date(2026, 10, 25)  # a Sunday
        self.assertEqual(len(receipts_in_period(receipts, "All time", today)), 4)
        self.assertEqual(
            [r["date"] for r in receipts_in_period(receipts, "This month", today)],
            ["2026-10-19", "2026-10-18"],
        )
        self.assertEqual(
            [r["date"] for r in receipts_in_period(receipts, "This week", today)],
            ["2026-10-19"],
        )


    def test_monthly_estimate_uses_weekly_rate(self):
        self.assertAlmostEqual(monthly_estimate(5.50, 5), 119.075)
        self.assertTrue(is_small_purchase(15))
        self.assertFalse(is_small_purchase(15.01))
        self.assertFalse(is_small_purchase(0))

    def test_saving_suggestion_halves_the_habit(self):
        coffee = saving_suggestion("Coffee & drinks", 5.50, 5)
        self.assertEqual(coffee["new_per_week"], 3)
        self.assertEqual(coffee["fewer_per_week"], 2)
        self.assertAlmostEqual(coffee["monthly_saving"], 47.63)
        self.assertIn("make it at home", coffee["tip"])

        weekly = saving_suggestion("Mystery", 10, 1)
        self.assertEqual(weekly["new_per_week"], 0.5)
        self.assertAlmostEqual(weekly["monthly_saving"], 21.65)
        self.assertIn("every other week", weekly["tip"])

        self.assertIn("The other 1 time,", saving_suggestion("Dining out", 12, 2)["tip"])


class StorageTests(unittest.TestCase):
    def test_receipts_round_trip_and_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data" / "receipts.json"
            self.assertEqual(load_receipts(path), [])
            receipts = [{"merchant": "Campus Cafe", "total": 6.25, "items": []}]
            save_receipts(receipts, path)
            self.assertEqual(load_receipts(path), receipts)

    def test_corrupt_file_raises_instead_of_returning_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipts.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(StorageError):
                load_receipts(path)


if __name__ == "__main__":
    unittest.main()
