import json
import re
from functools import lru_cache

from src.config import settings


def normalize_text(text: str) -> str:
    """Collapse PDF line breaks into spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_quotes(text: str) -> str:
    """Replace curly/smart quotes with straight apostrophes."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def load_dish_id_mapping() -> dict[str, int]:
    """Load dish_mapping.json: canonical name -> dish ID."""
    with open(settings.dish_mapping_path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_canonical_dish_names() -> dict[str, str]:
    """Load dish_mapping.json and return a normalized-name -> canonical-name lookup.

    Keys are normalized (whitespace-collapsed, straight quotes).
    Values are the original canonical names from the mapping file.
    """
    with open(settings.dish_mapping_path) as f:
        mapping: dict[str, int] = json.load(f)

    lookup: dict[str, str] = {}
    for canonical_name in mapping:
        normalized = normalize_quotes(normalize_text(canonical_name))
        lookup[normalized] = canonical_name
    return lookup
