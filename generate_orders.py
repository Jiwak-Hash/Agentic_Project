from datetime import date, timedelta
from pathlib import Path
import random

import pandas as pd


SEED = 42
ORDER_COUNT = 200
OUTPUT_PATH = Path(__file__).parent / "data" / "orders.csv"

PRODUCTS = [
    ("Wireless Earbuds", 2499),
    ("Cotton T-Shirt", 799),
    ("Insulated Water Bottle", 1099),
    ("Smart Watch", 4999),
    ("Bluetooth Speaker", 1899),
    ("Laptop Backpack", 1599),
    ("Ceramic Dinner Set", 2299),
    ("Running Shoes", 3199),
    ("USB-C Charger", 999),
    ("Desk Lamp", 1399),
]

PAYMENT_METHODS = ["UPI", "Card", "Cash on Delivery"]


def validate_orders(orders):
    required_columns = [
        "order_id",
        "customer_id",
        "product",
        "price",
        "status",
        "order_date",
        "delivery_date",
        "payment_method",
    ]
    missing_columns = set(required_columns) - set(orders.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    required_value_columns = [
        column for column in required_columns if column != "delivery_date"
    ]
    missing_values = orders[required_value_columns].isna().any()
    blank_values = orders[required_value_columns].astype(str).apply(
        lambda column: column.str.strip().eq("").any()
    )
    if missing_values.any() or blank_values.any():
        raise ValueError("Orders contain missing required values")

    if orders["order_id"].duplicated().any():
        raise ValueError("Order IDs must be unique")

    today = date.today()
    for row in orders.itertuples(index=False):
        order_date = date.fromisoformat(row.order_date)
        delivery_date_value = "" if pd.isna(row.delivery_date) else row.delivery_date
        if order_date > today:
            raise ValueError(f"Order {row.order_id} has a future order date")

        if row.status == "Delivered":
            if not delivery_date_value:
                raise ValueError(f"Delivered order {row.order_id} has no delivery date")
            delivery_date = date.fromisoformat(delivery_date_value)
            if not order_date < delivery_date <= today:
                raise ValueError(f"Order {row.order_id} has an inconsistent delivery date")
        elif delivery_date_value:
            raise ValueError(
                f"Non-delivered order {row.order_id} must have an empty delivery date"
            )


def generate_orders():
    random.seed(SEED)
    today = date.today()
    orders = []

    for order_id in range(1001, 1001 + ORDER_COUNT):
        days_since_order = random.randint(0, 180)
        order_date = today - timedelta(days=days_since_order)

        if days_since_order < 3:
            status = random.choices(
                ["Processing", "Shipped", "Cancelled"], weights=[75, 10, 15]
            )[0]
        elif days_since_order < 8:
            status = random.choices(
                ["Processing", "Shipped", "Delivered", "Cancelled", "Delayed"],
                weights=[25, 40, 15, 10, 10],
            )[0]
        else:
            status = random.choices(
                ["Processing", "Shipped", "Delivered", "Cancelled", "Delayed"],
                weights=[8, 20, 50, 12, 10],
            )[0]

        delivery_date = ""
        if status == "Delivered":
            delivery_days = random.randint(1, min(7, days_since_order))
            delivery_date = (order_date + timedelta(days=delivery_days)).isoformat()

        product, price = random.choice(PRODUCTS)
        orders.append(
            {
                "order_id": order_id,
                "customer_id": random.randint(1, 50),
                "product": product,
                "price": price,
                "status": status,
                "order_date": order_date.isoformat(),
                "delivery_date": delivery_date,
                "payment_method": random.choice(PAYMENT_METHODS),
            }
        )

    return pd.DataFrame(orders)


def main():
    orders = generate_orders()
    validate_orders(orders)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(orders)} orders to {OUTPUT_PATH}")
    print("\nFirst 10 orders:")
    print(orders.head(10).to_string(index=False))
    print("\nStatus distribution:")
    print(orders["status"].value_counts().sort_index().to_string())
    print("\nValidation passed: unique IDs, required values, and dates are consistent.")


if __name__ == "__main__":
    main()