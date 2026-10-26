"""Small, editable rules for categorizing receipt line items."""

import re
from datetime import date, timedelta
from typing import Any


# Each category's description is sent to the AI model to guide its choice.
CATEGORY_DESCRIPTIONS = {
    "Groceries": (
        "Food and ingredients bought to take home, including meat, produce, "
        "pantry staples, snacks, and ready-to-eat deli items such as rotisserie "
        "chicken from grocery or warehouse stores"
    ),
    "Dining out": (
        "Meals and food ordered from restaurants, cafes, fast food, food "
        "trucks, or delivery apps"
    ),
    "Coffee & drinks": (
        "Coffee, tea, boba, smoothies, and other drinks bought at a cafe or "
        "drink shop"
    ),
    "Household & paper products": (
        "Paper towels, toilet paper, tissues, trash bags, cleaning supplies, "
        "laundry detergent, dish soap, foil, and food storage"
    ),
    "Personal care & health": (
        "Toiletries, shampoo, toothpaste, skincare, cosmetics, vitamins, "
        "medicine, and first aid"
    ),
    "Clothing & accessories": "Clothes, shoes, bags, and accessories",
    "Electronics": "Devices, chargers, cables, headphones, and batteries",
    "School & office supplies": (
        "Notebooks, pens, paper, printing, textbooks, and other books"
    ),
    "Transportation": (
        "Gas, rideshare, transit fares, parking, tolls, and car maintenance"
    ),
    "Entertainment": "Movies, events, games, streaming, and hobbies",
    "Other": "Anything that does not clearly fit another category",
}

CATEGORIES = tuple(CATEGORY_DESCRIPTIONS)

# Keyword fallback for items the model leaves uncategorized. The first
# matching category wins, so more specific categories come first.
RULES = {
    "Household & paper products": (
        "paper towel",
        "toilet paper",
        "bath tissue",
        "tissue",
        "napkin",
        "trash bag",
        "detergent",
        "dish soap",
        "cleaner",
        "bleach",
        "foil",
        "plastic wrap",
    ),
    "Personal care & health": (
        "shampoo",
        "conditioner",
        "toothpaste",
        "deodorant",
        "lotion",
        "vitamin",
        "medicine",
        "ibuprofen",
        "bandage",
    ),
    "Coffee & drinks": (
        "coffee",
        "matcha",
        "latte",
        "espresso",
        "boba",
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
        "chkn",
        "chicken",
        "olive oil",
        "butter",
        "rice",
        "cereal",
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
    "Electronics": (
        "charger",
        "cable",
        "headphone",
        "earbud",
        "battery",
        "electronics",
    ),
    "School & office supplies": (
        "notebook",
        "pencil",
        "binder",
        "printing",
        "textbook",
        "book",
        "supplies",
    ),
    "Clothing & accessories": (
        "shirt",
        "pants",
        "jeans",
        "shoes",
        "sock",
        "clothing",
        "accessory",
    ),
    "Dining out": (
        "restaurant",
        "burger",
        "pizza",
        "sandwich",
        "taco",
        "meal",
        "dining",
        "food",
        "cafe",
        "steak",
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


# Category names used before the 11-category list, for receipts saved back then.
OLD_CATEGORY_NAMES = {"Drinks": "Coffee & drinks", "Food & restaurants": "Dining out"}


def current_category(category: str | None, name: str) -> str:
    """Map a saved item's category onto the current list."""
    if category in CATEGORIES:
        return category
    return OLD_CATEGORY_NAMES.get(category) or categorize_item(name)


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


PERIODS = ("All time", "This month", "This week")


def receipt_date(receipt: dict[str, Any]) -> date | None:
    try:
        return date.fromisoformat(receipt.get("date") or "")
    except ValueError:
        return None


def receipts_in_period(
    receipts: list[dict[str, Any]], period: str, today: date
) -> list[dict[str, Any]]:
    """Keep receipts dated in this calendar month or this Monday-to-Sunday week."""
    if period == "This month":
        start = today.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    elif period == "This week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
    else:
        return list(receipts)
    return [
        receipt
        for receipt in receipts
        if (day := receipt_date(receipt)) is not None and start <= day <= end
    ]


def line_item_sum(items: list[dict[str, Any]]) -> float | None:
    if not items:
        return None
    amounts = [amount_for_item(item) for item in items]
    if any(amount is None for amount in amounts):
        return None
    return sum(amount for amount in amounts if amount is not None)
