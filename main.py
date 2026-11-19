"""Independent CLI; --customer-id simulates a trusted session for this demo."""

import argparse
from datetime import date

from workflow import build_workflow


def main():
    parser = argparse.ArgumentParser(description="DemoCart support workflow (local demo session, not authentication).")
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument("--evaluation-date", type=date.fromisoformat)
    parser.add_argument("message")
    args = parser.parse_args()
    try:
        result = build_workflow().invoke({
            "message": args.message, "customer_id": args.customer_id,
            "evaluation_date": args.evaluation_date,
        })
    except ValueError as error:
        parser.error(str(error))
    print(f"Intent: {result['intent']} | Destination: {result['selected_agent']}")
    print(result["response"])
    if result["sources"]:
        print("Sources: " + ", ".join(result["sources"]))
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
