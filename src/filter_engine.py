import re

import pandas as pd

from src.knowledge import get_planets_within, get_techniques_in_category


def _parse_grade(grade_str: str) -> int:
    """Convert a license grade string to an integer for comparison.

    Handles Roman numerals (I-VI, VI+), Arabic numbers, and mixed formats.
    """
    grade_str = grade_str.strip().upper()

    roman_map = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI+": 7, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    }
    if grade_str in roman_map:
        return roman_map[grade_str]

    digits = re.sub(r"[^\d]", "", grade_str)
    if digits:
        return int(digits)

    return 0


def apply_filters(df: pd.DataFrame, filters: dict) -> list[int]:
    """Apply parsed filters to the dishes DataFrame, return matching dish IDs."""
    mask = pd.Series(True, index=df.index)

    # --- Ingredients include (AND) ---
    if filters.get("ingredients_include"):
        for ing in filters["ingredients_include"]:
            mask &= df["ingredients"].apply(lambda x: ing in x)

    # --- Ingredients exclude (NOT) ---
    if filters.get("ingredients_exclude"):
        for ing in filters["ingredients_exclude"]:
            mask &= df["ingredients"].apply(lambda x: ing not in x)

    # --- Techniques include (AND) ---
    if filters.get("techniques_include"):
        for tech in filters["techniques_include"]:
            mask &= df["techniques"].apply(lambda x: tech in x)

    # --- Techniques exclude (NOT) ---
    if filters.get("techniques_exclude"):
        for tech in filters["techniques_exclude"]:
            mask &= df["techniques"].apply(lambda x: tech not in x)

    # --- Ingredients any (OR) ---
    if filters.get("ingredients_any"):
        candidates = filters["ingredients_any"]
        mask &= df["ingredients"].apply(
            lambda x: any(ing in x for ing in candidates)
        )

    # --- Techniques any (OR) ---
    if filters.get("techniques_any"):
        candidates = filters["techniques_any"]
        mask &= df["techniques"].apply(
            lambda x: any(tech in x for tech in candidates)
        )

    # --- Min ingredients from (at least N of M) ---
    mif = filters.get("min_ingredients_from")
    if mif and mif.get("candidates") and mif.get("min_count"):
        candidates = mif["candidates"]
        min_count = mif["min_count"]
        mask &= df["ingredients"].apply(
            lambda x: sum(1 for ing in candidates if ing in x) >= min_count
        )

    # --- Restaurant ---
    if filters.get("restaurant"):
        mask &= df["restaurant"] == filters["restaurant"]

    # --- Planet ---
    if filters.get("planet"):
        mask &= df["planet"] == filters["planet"]

    # --- License filter ---
    lf = filters.get("license_filter")
    if lf and lf.get("type") and lf.get("min_grade") is not None:
        lic_type = lf["type"]
        min_grade = lf["min_grade"]

        def _has_license(licenses: list[dict]) -> bool:
            for lic in licenses:
                if lic["type"].lower() == lic_type.lower():
                    return _parse_grade(lic["grade"]) >= min_grade
            return False

        mask &= df["licenses"].apply(_has_license)

    # --- Technique categories include ---
    if filters.get("technique_categories_include"):
        category_techniques = set()
        for cat in filters["technique_categories_include"]:
            category_techniques.update(get_techniques_in_category(cat))

        if category_techniques:
            mask &= df["techniques"].apply(
                lambda x: bool(set(x) & category_techniques)
            )

    # --- Distance constraint ---
    dc = filters.get("distance_constraint")
    if dc and dc.get("origin") and dc.get("max_ly") is not None:
        origin = dc["origin"]
        max_ly = dc["max_ly"]
        valid_planets = set(get_planets_within(origin, max_ly))
        mask &= df["planet"].isin(valid_planets)

    matched = df.loc[mask, "dish_id"].dropna().astype(int).tolist()
    return sorted(matched)
