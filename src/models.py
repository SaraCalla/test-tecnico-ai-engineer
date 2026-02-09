from pydantic import BaseModel


class License(BaseModel):
    """A chef's certified skill/license."""

    type: str  # e.g. "Psionica", "Temporale", "Quantistica"
    grade: str  # e.g. "II", "13", "I" — kept as string for flexibility


class Dish(BaseModel):
    """A single dish from a menu."""

    name: str
    ingredients: list[str]
    techniques: list[str]


class MenuData(BaseModel):
    """Structured data extracted from one restaurant menu PDF."""

    restaurant: str
    chef: str
    planet: str
    licenses: list[License]
    ltk: str  # e.g. "II", "IX", "VI+"
    dishes: list[Dish]
    source_file: str


class DishValidationResult(BaseModel):
    """Result of validating a single dish name against canonical names."""

    original_name: str
    canonical_name: str | None = None
    match_type: str  # "exact", "normalized", "fuzzy", "unmatched"
    similarity_score: float
    candidates: list[tuple[str, float]] = []


class MenuValidationReport(BaseModel):
    """Aggregated validation report for one menu."""

    source_file: str
    total_dishes: int
    exact_matches: int
    normalized_matches: int
    fuzzy_matches: int
    unmatched: int
    details: list[DishValidationResult]
