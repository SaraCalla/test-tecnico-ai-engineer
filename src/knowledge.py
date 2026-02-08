import json

import pandas as pd
from pypdf import PdfReader

from src.config import settings
from src.logger import logger
from src.parsing import parse_json_response


# ── Planet distances ───────────────────────────────────────────────


def load_distance_matrix() -> pd.DataFrame:
    """Load the planet distance matrix from Distanze.csv.

    Returns a DataFrame indexed by planet name with planet columns.
    """
    df = pd.read_csv(settings.distances_csv_path, index_col=0)
    df.index.name = "planet"
    return df


def get_distance(planet_a: str, planet_b: str) -> int | None:
    """Return the distance in light-years between two planets, or None if unknown."""
    df = load_distance_matrix()
    try:
        return int(df.loc[planet_a, planet_b])
    except KeyError:
        return None


def get_planets_within(origin: str, max_distance: int) -> list[str]:
    """Return all planets within max_distance light-years of origin."""
    df = load_distance_matrix()
    if origin not in df.index:
        return []
    distances = df.loc[origin]
    return [planet for planet, dist in distances.items() if int(dist) <= max_distance]


# ── Technique categories from "Manuale di Cucina" ──────────────────

_TECHNIQUE_EXTRACTION_PROMPT = """\
Extract ALL technique categories and their techniques from this cooking manual.

Return a JSON object where:
- Each key is a technique CATEGORY name in lowercase (e.g. "marinatura", "taglio", "bollitura")
- Each value is a list of the EXACT technique names that belong to that category

Include ALL categories and ALL techniques from the manual. Do not skip any.
Return ONLY valid JSON, no markdown fences, no commentary.

Manual text:
"""


def _extract_technique_categories_from_pdf() -> dict[str, list[str]]:
    """Use the LLM to extract technique categories from Manuale di Cucina."""
    from datapizza.clients.openai import OpenAIClient

    reader = PdfReader(settings.manuale_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    client = OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=0.0,
    )
    response = client.invoke(_TECHNIQUE_EXTRACTION_PROMPT + text)

    return parse_json_response(response.text)


def load_technique_categories(
    cache_path: str | None = None,
) -> dict[str, list[str]]:
    """Load technique categories, extracting from PDF if not cached.

    Returns a dict: category name (lowercase) -> list of technique names.
    """
    if cache_path is None:
        cache_path = settings.output_dir / "technique_categories.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    logger.info("Extracting technique categories from Manuale di Cucina...")
    categories = _extract_technique_categories_from_pdf()

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    logger.info(f"  Cached {len(categories)} categories to {cache_path}")

    return categories


def get_techniques_in_category(
    category: str,
    categories: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return all technique names belonging to a category (case-insensitive)."""
    if categories is None:
        categories = load_technique_categories()
    return categories.get(category.lower(), [])


def get_all_categories(
    categories: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return all known technique category names."""
    if categories is None:
        categories = load_technique_categories()
    return list(categories.keys())
