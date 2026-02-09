import json
import re
from pathlib import Path

from pypdf import PdfReader

from src.config import settings
from src.logger import logger
from src.models import MenuData, MenuValidationReport
from src.normalization import load_canonical_dish_names, load_dish_id_mapping
from src.validation import _log_validation_summary, _validate_and_report, validate_menu_dishes

# Backward-compatible re-exports: these names were historically imported
# from src.parsing by other modules. Keep them importable from here.
# __all__ = [
#     "MenuData",
#     "load_dish_id_mapping",
#     "parse_json_response",
#     "parse_all_menus",
#     "parse_menu_pdf",
# ]


# ── LLM client ──────────────────────────────────────────────────────


def _get_llm_client():
    """Lazy-initialize the OpenAI client."""
    from datapizza.clients.openai import OpenAIClient

    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse LLM response as JSON: {e}. "
            f"Response text (first 200 chars): {text[:200]!r}"
        ) from e


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


def build_extraction_prompt(dish_names: list[str]) -> str:
    """Build the full extraction prompt with canonical dish names injected.

    Args:
        dish_names: List of canonical dish names to include in the prompt.
    """
    dish_name_list = "\n".join(f"- {name}" for name in dish_names)
    return _MENU_EXTRACTION_PROMPT_TEMPLATE.format(dish_name_list=dish_name_list)


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

    canonical = load_canonical_dish_names()
    dish_names = sorted(set(canonical.values()))

    raw_text = extract_pdf_text(pdf_path)
    prompt = build_extraction_prompt(dish_names) + raw_text
    response = client.invoke(prompt)
    data = parse_json_response(response.text)
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
        logger.info(f"Loading existing cache from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        cached_menus = {m["source_file"]: m for m in data}
        logger.info(f"  Found {len(cached_menus)} cached menus")

    # Identify which PDFs need parsing
    pdf_files = sorted(menu_dir.glob("*.pdf"))
    to_parse = [p for p in pdf_files if p.name not in cached_menus]

    if not to_parse:
        logger.info("All menus already cached!")
        menus = [MenuData(**m) for m in cached_menus.values()]
        _validate_and_report(menus)
        return menus

    logger.info(f"Need to parse {len(to_parse)}/{len(pdf_files)} menus\n")

    # Parse missing PDFs with incremental saving
    client = _get_llm_client()

    for i, pdf_path in enumerate(to_parse):
        logger.info(f"  [{i + 1}/{len(to_parse)}] Parsing {pdf_path.name}...")
        try:
            menu, report = parse_menu_pdf(pdf_path, client=client)
            cached_menus[pdf_path.name] = menu.model_dump()
            logger.info(f"    -> {len(menu.dishes)} dishes, planet={menu.planet}")

            if report:
                _log_validation_summary(report)

            # Save incrementally after each successful parse
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_path, "w") as f:
                json.dump(
                    list(cached_menus.values()),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(f"    Saved to cache ({len(cached_menus)}/{len(pdf_files)} total)")

        except Exception as e:
            logger.error(f"    FAILED: {e}")

    logger.info(f"\nParsing complete! {len(cached_menus)} menus in cache")
    return [MenuData(**m) for m in cached_menus.values()]
