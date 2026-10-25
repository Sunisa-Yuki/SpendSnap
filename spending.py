"""Small, editable rules for categorizing receipt line items."""

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
        "combo",
    ),
}

def categorize_item(name: str) -> str:
    normalized = name.casefold()
    for category, keywords in RULES.items():
        if any(keyword in normalized for keyword in keywords):
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
