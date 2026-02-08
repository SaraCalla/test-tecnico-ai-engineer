import difflib
import json
import re

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from src.config import settings


# ── Data models ──────────────────────────────────────────────────────


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


# ── Text normalization ───────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Collapse PDF line breaks into spaces."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_quotes(text: str) -> str:
    """Replace curly/smart quotes with straight apostrophes."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


# ── Canonical dish names ────────────────────────────────────────────


def load_dish_id_mapping() -> dict[str, int]:
    """Load dish_mapping.json: canonical name -> dish ID."""
    with open(settings.dish_mapping_path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_canonical_dish_names() -> dict[str, str]:
    """Load dish_mapping.json and return a normalized-name → canonical-name lookup.

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


# ── LLM client ───────────────────────────────────────────────────────


def _get_llm_client():
    """Lazy-initialize the OpenAI client."""
    from datapizza.clients.openai import OpenAIClient

    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


# ── PDF text extraction ──────────────────────────────────────────────


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── Menu extraction ──────────────────────────────────────────────────

_MENU_EXTRACTION_PROMPT_TEMPLATE = """\
Extract structured data from this restaurant menu.

Return a JSON object with EXACTLY this structure:
{{
  "restaurant": "restaurant name (without surrounding quotes)",
  "chef": "chef full name",
  "planet": "planet where the restaurant is located",
  "licenses": [{{"type": "license category name", "grade": "level as string"}}],
  "ltk": "LTK level as string (e.g. II, IX, VI+)",
  "dishes": [
    {{
      "name": "canonical dish name from the reference list below",
      "ingredients": ["ingredient 1", "ingredient 2"],
      "techniques": ["technique 1", "technique 2"]
    }}
  ]
}}

IMPORTANT RULES:
1. Extract ALL dishes — do not skip any
2. For licenses/skills, normalize the type name to the standard category:
   - "EDUCATION di livello 11" with context about Quantistica → type="Quantistica", grade="11"
   - "Education Level Temporale II" → type="Temporale", grade="II"
   - "Psionica II" → type="Psionica", grade="II"
   The standard categories are: Psionica, Temporale, Gravitazionale, Antimateria, Magnetica, Quantistica, Luce
3. The planet is usually mentioned in the chef's bio narrative
4. For narrative-style menus (ingredients/techniques embedded in prose, no bullet lists),
   carefully identify ALL ingredients and techniques mentioned in the text for each dish
5. For dish names: use ONLY the CANONICAL NAME from the reference list below when the dish
   appears in this menu. Match by meaning/similarity, not character-for-character.
   Use the FULL canonical name — never truncate or shorten it.
   Only include dishes that actually appear in THIS menu — do NOT invent dishes.
6. Do NOT include emoji characters in dish names
7. Return ONLY valid JSON, no markdown fences, no commentary

CANONICAL DISH NAME REFERENCE LIST:
{dish_name_list}

Menu text:
"""


def build_extraction_prompt(dish_names: list[str] | None = None) -> str:
    """Build the full extraction prompt with canonical dish names injected."""
    if dish_names is None:
        canonical = load_canonical_dish_names()
        dish_names = sorted(set(canonical.values()))

    dish_name_list = "\n".join(f"- {name}" for name in dish_names)
    return _MENU_EXTRACTION_PROMPT_TEMPLATE.format(dish_name_list=dish_name_list)


# ── Dish name validation ────────────────────────────────────────────

FUZZY_THRESHOLD = 0.85  # minimum SequenceMatcher ratio to auto-correct


@dataclass
class DishValidationResult:
    """Result of validating a single dish name against canonical names."""

    original_name: str
    canonical_name: str | None  # None if no match found
    match_type: str  # "exact", "normalized", "fuzzy", "unmatched"
    similarity_score: float  # 1.0 for exact, 0.0-1.0 for fuzzy
    candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class MenuValidationReport:
    """Aggregated validation report for one menu."""

    source_file: str
    total_dishes: int
    exact_matches: int
    normalized_matches: int
    fuzzy_matches: int
    unmatched: int
    details: list[DishValidationResult]


def validate_dish_name(
    extracted_name: str,
    canonical_lookup: dict[str, str],
) -> DishValidationResult:
    """Validate a single extracted dish name against canonical names.

    Matching strategy (in order):
    1. Exact match against canonical names
    2. Normalized match (whitespace + quote normalization)
    3. Fuzzy match using difflib.SequenceMatcher
    4. Unmatched — return original with candidate suggestions
    """
    canonical_names = set(canonical_lookup.values())

    # 1. Exact match
    if extracted_name in canonical_names:
        return DishValidationResult(
            original_name=extracted_name,
            canonical_name=extracted_name,
            match_type="exact",
            similarity_score=1.0,
        )

    # 2. Normalized match
    norm_extracted = normalize_quotes(normalize_text(extracted_name))
    if norm_extracted in canonical_lookup:
        canonical = canonical_lookup[norm_extracted]
        return DishValidationResult(
            original_name=extracted_name,
            canonical_name=canonical,
            match_type="normalized",
            similarity_score=1.0,
        )

    # 3. Fuzzy match
    candidates = difflib.get_close_matches(
        norm_extracted,
        canonical_lookup.keys(),
        n=3,
        cutoff=0.6,
    )

    scored_candidates = [
        (canonical_lookup[c], difflib.SequenceMatcher(None, norm_extracted, c).ratio())
        for c in candidates
    ]

    if scored_candidates and scored_candidates[0][1] >= FUZZY_THRESHOLD:
        best_name, best_score = scored_candidates[0]
        return DishValidationResult(
            original_name=extracted_name,
            canonical_name=best_name,
            match_type="fuzzy",
            similarity_score=best_score,
            candidates=scored_candidates,
        )

    # 4. Unmatched
    return DishValidationResult(
        original_name=extracted_name,
        canonical_name=None,
        match_type="unmatched",
        similarity_score=scored_candidates[0][1] if scored_candidates else 0.0,
        candidates=scored_candidates,
    )


def validate_menu_dishes(
    menu: MenuData,
    canonical_lookup: dict[str, str] | None = None,
) -> MenuValidationReport:
    """Validate all dish names in a parsed menu against canonical names.

    Auto-corrects dish names in-place for exact/normalized/fuzzy matches.
    Returns a validation report with details.
    """
    if canonical_lookup is None:
        canonical_lookup = load_canonical_dish_names()

    details: list[DishValidationResult] = []

    for dish in menu.dishes:
        result = validate_dish_name(dish.name, canonical_lookup)
        details.append(result)

        # Auto-correct for matched names
        if result.canonical_name is not None:
            dish.name = result.canonical_name

    return MenuValidationReport(
        source_file=menu.source_file,
        total_dishes=len(menu.dishes),
        exact_matches=sum(1 for d in details if d.match_type == "exact"),
        normalized_matches=sum(1 for d in details if d.match_type == "normalized"),
        fuzzy_matches=sum(1 for d in details if d.match_type == "fuzzy"),
        unmatched=sum(1 for d in details if d.match_type == "unmatched"),
        details=details,
    )


# ── Validation reporting ────────────────────────────────────────────


def _print_validation_summary(report: MenuValidationReport) -> None:
    """Print a single-menu validation summary (only if issues found)."""
    if report.unmatched == 0 and report.fuzzy_matches == 0:
        return

    parts = []
    if report.fuzzy_matches > 0:
        parts.append(f"{report.fuzzy_matches} fuzzy-corrected")
    if report.unmatched > 0:
        parts.append(f"{report.unmatched} UNMATCHED")
    print(f"    [validation] {', '.join(parts)}")

    for d in report.details:
        if d.match_type == "fuzzy":
            print(
                f"      ~ '{d.original_name}' -> '{d.canonical_name}' "
                f"(score={d.similarity_score:.2f})"
            )
        elif d.match_type == "unmatched":
            print(f"      ! '{d.original_name}' has NO canonical match")
            for name, score in d.candidates[:3]:
                print(f"        candidate: '{name}' (score={score:.2f})")


def _validate_and_report(menus: list[MenuData]) -> None:
    """Validate all menus loaded from cache and print a summary."""
    canonical_lookup = load_canonical_dish_names()
    total_unmatched = 0
    total_fuzzy = 0

    for menu in menus:
        report = validate_menu_dishes(menu, canonical_lookup)
        total_unmatched += report.unmatched
        total_fuzzy += report.fuzzy_matches
        if report.unmatched > 0 or report.fuzzy_matches > 0:
            _print_validation_summary(report)

    if total_unmatched == 0 and total_fuzzy == 0:
        print("  [validation] All cached dish names match canonical names")
    else:
        print(
            f"  [validation] Summary: {total_fuzzy} fuzzy corrections, "
            f"{total_unmatched} unmatched across all cached menus"
        )


def _print_overall_validation_summary(
    reports: list[MenuValidationReport],
) -> None:
    """Print aggregate validation statistics across freshly parsed menus."""
    total = sum(r.total_dishes for r in reports)
    exact = sum(r.exact_matches for r in reports)
    normalized = sum(r.normalized_matches for r in reports)
    fuzzy = sum(r.fuzzy_matches for r in reports)
    unmatched = sum(r.unmatched for r in reports)

    print("\n--- Validation Summary ---")
    print(f"  Total dishes:        {total}")
    print(f"  Exact matches:       {exact}")
    print(f"  Normalized matches:  {normalized}")
    print(f"  Fuzzy corrections:   {fuzzy}")
    print(f"  Unmatched:           {unmatched}")
    if unmatched > 0:
        print(
            f"  WARNING: {unmatched} dish(es) could not be matched "
            "to any canonical name"
        )


# ── Parsing functions ───────────────────────────────────────────────


def parse_menu_pdf(
    pdf_path: Path,
    client=None,
    validate: bool = True,
) -> tuple[MenuData, MenuValidationReport | None]:
    """Parse a single menu PDF into structured data using LLM extraction.

    Returns:
        A tuple of (MenuData, MenuValidationReport or None).
    """
    if client is None:
        client = _get_llm_client()

    raw_text = extract_pdf_text(pdf_path)
    prompt = build_extraction_prompt() + raw_text
    response = client.invoke(prompt)
    data = _parse_json_response(response.text)
    data["source_file"] = pdf_path.name

    menu = MenuData(**data)

    report = None
    if validate:
        report = validate_menu_dishes(menu)

    return menu, report


def parse_all_menus(
    menu_dir: Path | None = None,
    cache_path: Path | None = None,
) -> list[MenuData]:
    """Parse all menu PDFs with incremental caching.

    Supports resuming: if cache exists, only parses missing PDFs and saves after each one.

    Args:
        menu_dir: Directory containing menu PDFs.
        cache_path: Where to save/load the parsed JSON cache.
    """
    menu_dir = menu_dir or settings.menu_dir
    cache_path = cache_path or settings.output_dir / "parsed_menus.json"

    # Load existing cache (if any)
    cached_menus: dict[str, dict] = {}
    if cache_path.exists():
        print(f"Loading existing cache from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        cached_menus = {m["source_file"]: m for m in data}
        print(f"  Found {len(cached_menus)} cached menus")

    # Identify which PDFs need parsing
    pdf_files = sorted(menu_dir.glob("*.pdf"))
    to_parse = [p for p in pdf_files if p.name not in cached_menus]

    if not to_parse:
        print("All menus already cached!")
        menus = [MenuData(**m) for m in cached_menus.values()]
        _validate_and_report(menus)
        return menus

    print(f"Need to parse {len(to_parse)}/{len(pdf_files)} menus\n")

    # Parse missing PDFs with incremental saving
    client = _get_llm_client()

    for i, pdf_path in enumerate(to_parse):
        print(f"  [{i + 1}/{len(to_parse)}] Parsing {pdf_path.name}...")
        try:
            menu, report = parse_menu_pdf(pdf_path, client=client)
            cached_menus[pdf_path.name] = menu.model_dump()
            print(f"    -> {len(menu.dishes)} dishes, planet={menu.planet}")

            if report:
                _print_validation_summary(report)

            # Save incrementally after each successful parse
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_path, "w") as f:
                json.dump(
                    list(cached_menus.values()),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"    Saved to cache ({len(cached_menus)}/{len(pdf_files)} total)")

        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nParsing complete! {len(cached_menus)} menus in cache")
    return [MenuData(**m) for m in cached_menus.values()]


