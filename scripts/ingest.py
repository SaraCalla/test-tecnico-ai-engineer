"""One-time ingestion: parse all menu PDFs into structured data (cached)."""

from src.parsing import parse_all_menus


def main():
    # Parse all menus (cached after first run)
    print("=== Parsing menus ===")
    menus = parse_all_menus()
    print(f"Parsed {len(menus)} menus, {sum(len(m.dishes) for m in menus)} total dishes")


if __name__ == "__main__":
    main()
