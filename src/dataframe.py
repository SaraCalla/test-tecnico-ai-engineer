import pandas as pd

from src.logger import logger
from src.parsing import MenuData, load_dish_id_mapping, parse_all_menus


def build_dishes_dataframe(menus: list[MenuData] | None = None) -> pd.DataFrame:
    """Build a DataFrame with one row per dish, inheriting restaurant metadata.

    Columns: dish_id, dish_name, restaurant, planet, chef, ingredients,
    techniques, licenses, ltk, source_file.
    """
    if menus is None:
        menus = parse_all_menus()

    dish_id_mapping = load_dish_id_mapping()
    rows = []

    for menu in menus:
        for dish in menu.dishes:
            rows.append({
                "dish_id": dish_id_mapping.get(dish.name),
                "dish_name": dish.name,
                "restaurant": menu.restaurant,
                "planet": menu.planet,
                "chef": menu.chef,
                "ingredients": dish.ingredients,
                "techniques": dish.techniques,
                "licenses": [
                    {"type": lic.type, "grade": lic.grade}
                    for lic in menu.licenses
                ],
                "ltk": menu.ltk,
                "source_file": menu.source_file,
            })

    df = pd.DataFrame(rows)
    unmatched = df["dish_id"].isna().sum()
    if unmatched > 0:
        logger.warning(f"  {unmatched} dishes have no ID in dish_mapping.json")

    return df
