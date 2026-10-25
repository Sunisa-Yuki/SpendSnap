"""Small, editable rules for categorizing receipt line items."""

import re
from typing import Any


CATEGORIES = (
    "Drinks",
    "Food & restaurants",
    "Groceries",
    "Shopping",
    "Transportation",
    "Other",
)

RULES = {
    "Drinks": (
        "coffee",
        "matcha",
        "tea",
        "latte",
        "espresso",
        "smoothie",
        "juice",
        "soda",
        "drink",
        "beverage",
    ),
    "Groceries": (
        "grocery",
        "produce",
        "milk",
        "bread",
        "egg",
        "tomato",
        "vegetable",
        "fruit",
        "market",
        "supermarket",
    ),
    "Transportation": (
        "uber",
        "lyft",
        "transit",
        "bus",
        "train",
        "parking",
        "gas",
        "fuel",
        "fare",
    ),
    "Shopping": (
        "shirt",
        "clothing",
        "book",
        "supplies",
        "accessory",
        "charger",
        "electronics",
    ),
    "Food & restaurants": (
        "restaurant",
        "burger",
        "pizza",
        "sandwich",
        "taco",
        "salad",
        "meal",
        "dining",
        "food",
        "cafe",
        "chicken",
        "chkn",
        "beef",
        "steak",
        "pork",
        "fish",
        "shrimp",
        "rice",
        "noodle",
        "ramen",
        "pasta",
        "sushi",
        "curry",
        "bowl",
        "wrap",
        "burrito",
        "fries",
        "wing",
        "dumpling",
        "soup",
        "combo",
        "entree",
    ),
}


def categorize_item(name: str) -> str:
    """Fallback when the AI gives no category; keywords match at word starts only."""
    normalized = name.casefold()
    for category, keywords in RULES.items():
        if any(re.search(rf"\b{re.escape(keyword)}", normalized) for keyword in keywords):
            return category
    return "Other"


def amount_for_item(item: dict[str, Any]) -> float | None:
    """Return a known line total, or derive it only from known price and quantity."""
    total = item.get("line_total")
    if total is not None:
        return float(total)
    unit_price = item.get("unit_price")
    quantity = item.get("quantity")
    if unit_price is not None and quantity is not None:
        return float(unit_price) * float(quantity)
    return None


def line_item_sum(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    amounts = [amount_for_item(item) for item in items]
    if any(amount is None for amount in amounts):
        return None
    return sum(amount for amount in amounts if amount is not None)
