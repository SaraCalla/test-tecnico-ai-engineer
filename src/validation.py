import difflib

from src.logger import logger
from src.models import DishValidationResult, MenuData, MenuValidationReport
from src.normalization import load_canonical_dish_names, normalize_quotes, normalize_text

# Two thresholds are used for fuzzy matching:
# - FUZZY_CANDIDATE_CUTOFF (loose): gathers plausible candidates so that unmatched
#   dishes still show nearby suggestions in the validation log for debugging.
# - FUZZY_THRESHOLD (strict): only candidates above this score are confident enough
#   to auto-correct the dish name without human review.
FUZZY_THRESHOLD = 0.85
FUZZY_CANDIDATE_CUTOFF = 0.6


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
        cutoff=FUZZY_CANDIDATE_CUTOFF,
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


def _log_validation_summary(report: MenuValidationReport) -> None:
    """Log a single-menu validation summary (only if issues found)."""
    if report.unmatched == 0 and report.fuzzy_matches == 0:
        return

    parts = []
    if report.fuzzy_matches > 0:
        parts.append(f"{report.fuzzy_matches} fuzzy-corrected")
    if report.unmatched > 0:
        parts.append(f"{report.unmatched} UNMATCHED")
    logger.info(f"    [validation] {', '.join(parts)}")

    for d in report.details:
        if d.match_type == "fuzzy":
            logger.info(
                f"      ~ '{d.original_name}' -> '{d.canonical_name}' "
                f"(score={d.similarity_score:.2f})"
            )
        elif d.match_type == "unmatched":
            logger.warning(f"      ! '{d.original_name}' has NO canonical match")
            for name, score in d.candidates[:3]:
                logger.info(f"        candidate: '{name}' (score={score:.2f})")


def _validate_and_report(menus: list[MenuData]) -> None:
    """Validate all menus loaded from cache and log a summary."""
    canonical_lookup = load_canonical_dish_names()
    total_unmatched = 0
    total_fuzzy = 0

    for menu in menus:
        report = validate_menu_dishes(menu, canonical_lookup)
        total_unmatched += report.unmatched
        total_fuzzy += report.fuzzy_matches
        if report.unmatched > 0 or report.fuzzy_matches > 0:
            _log_validation_summary(report)

    if total_unmatched == 0 and total_fuzzy == 0:
        logger.info("  [validation] All cached dish names match canonical names")
    else:
        logger.info(
            f"  [validation] Summary: {total_fuzzy} fuzzy corrections, "
            f"{total_unmatched} unmatched across all cached menus"
        )
