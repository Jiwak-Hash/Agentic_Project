from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_DIR = Path(__file__).parent
CSV_PATH = PROJECT_DIR / "data" / "orders.csv"
DATABASE_PATH = PROJECT_DIR / "data" / "ecommerce.db"


def connect(database_path=DATABASE_PATH):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def create_orders_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            delivery_date TEXT,
            payment_method TEXT NOT NULL
        )
        """
    )


def import_orders(csv_path=CSV_PATH, database_path=DATABASE_PATH):
    orders = pd.read_csv(csv_path)
    connection = connect(database_path)
    try:
        create_orders_table(connection)
        rows = [
            (
                int(row.order_id),
                int(row.customer_id),
                row.product,
                float(row.price),
                row.status,
                row.order_date,
                None if pd.isna(row.delivery_date) else row.delivery_date,
                row.payment_method,
            )
            for row in orders.itertuples(index=False)
        ]
        cursor = connection.executemany(
            """
            INSERT OR IGNORE INTO orders (
                order_id, customer_id, product, price, status, order_date,
                delivery_date, payment_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
        return cursor.rowcount
    finally:
        connection.close()


def get_order(order_id, customer_id, database_path=DATABASE_PATH):
    connection = connect(database_path)
    try:
        row = connection.execute(
            "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
            (order_id, customer_id),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def get_customer_orders(customer_id, database_path=DATABASE_PATH):
    connection = connect(database_path)
    try:
        rows = connection.execute(
            "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_id",
            (customer_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def main():
    imported_count = import_orders()
    print(f"Database ready at {DATABASE_PATH}")
    print(f"Imported {imported_count} new orders; existing order IDs were skipped.")


if __name__ == "__main__":
    main()