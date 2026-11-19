import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

import database


PROJECT_DIR = Path(__file__).parent
ORDERS_CSV_PATH = PROJECT_DIR / "data" / "orders.csv"


class OrderDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls.temp_directory.name) / "test_ecommerce.db"
        database.import_orders(ORDERS_CSV_PATH, cls.database_path)
        with ORDERS_CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
            cls.first_order = next(csv.DictReader(csv_file))

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()

    def test_database_contains_200_orders(self):
        with sqlite3.connect(self.database_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        self.assertEqual(count, 200)

    def test_existing_order_can_be_retrieved_by_its_customer(self):
        order = database.get_order(
            int(self.first_order["order_id"]),
            int(self.first_order["customer_id"]),
            self.database_path,
        )
        self.assertIsNotNone(order)
        self.assertEqual(order["product"], self.first_order["product"])

    def test_unknown_order_returns_none(self):
        order = database.get_order(999999, 1, self.database_path)
        self.assertIsNone(order)

    def test_customer_cannot_retrieve_another_customers_order(self):
        other_customer_id = int(self.first_order["customer_id"]) % 50 + 1
        order = database.get_order(
            int(self.first_order["order_id"]), other_customer_id, self.database_path
        )
        self.assertIsNone(order)

    def test_repeated_import_does_not_create_duplicates(self):
        imported_count = database.import_orders(ORDERS_CSV_PATH, self.database_path)
        self.assertEqual(imported_count, 0)
        with sqlite3.connect(self.database_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        self.assertEqual(count, 200)

    def test_customer_order_list_only_contains_that_customer(self):
        customer_id = int(self.first_order["customer_id"])
        orders = database.get_customer_orders(customer_id, self.database_path)
        self.assertTrue(orders)
        self.assertTrue(all(order["customer_id"] == customer_id for order in orders))


if __name__ == "__main__":
    unittest.main()