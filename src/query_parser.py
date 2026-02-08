import pandas as pd
from datapizza.clients.openai import OpenAIClient

from src.config import settings
from src.knowledge import load_technique_categories
from src.parsing import parse_json_response

_PARSER_PROMPT_TEMPLATE = """\
You are a query parser for an intergalactic restaurant database.
Convert the user's question into a structured JSON filter.

Return a JSON object with EXACTLY this structure (use null for unused fields):
{{
  "ingredients_include": ["X", "Y"],
  "ingredients_exclude": ["Z"],
  "techniques_include": ["A"],
  "techniques_exclude": ["B"],
  "ingredients_any": ["X", "Y"],
  "techniques_any": ["A", "B"],
  "min_ingredients_from": {{"candidates": ["X", "Y", "Z"], "min_count": 2}},
  "restaurant": "name or null",
  "planet": "name or null",
  "license_filter": {{"type": "Psionica", "min_grade": 2}},
  "technique_categories_include": ["taglio"],
  "distance_constraint": {{"origin": "Namecc", "max_ly": 659}}
}}

FIELD EXPLANATIONS:
- ingredients_include: ALL these ingredients must be present (AND)
- ingredients_exclude: NONE of these ingredients must be present (NOT)
- techniques_include: ALL these techniques must be used (AND)
- techniques_exclude: NONE of these techniques must be used (NOT)
- ingredients_any: AT LEAST ONE of these must be present (OR)
- techniques_any: AT LEAST ONE of these must be used (OR)
- min_ingredients_from: at least min_count of the candidates must be present
- restaurant: filter by exact restaurant name
- planet: filter by planet where the restaurant is located
- license_filter: chef must have this license type at this minimum grade
  License types and their abbreviations:
    P = Psionica, t = Temporale, G = Gravitazionale, e+ = Antimateria,
    Mx = Magnetica, Q = Quantistica, c = Luce, LTK = Livello Tecnologico
  Always use the FULL type name (e.g. "Psionica", not "P").
  Convert grades to integers: I=1, II=2, III=3, IV=4, V=5, VI=6, VI+=7.
  "superiore a 0" means min_grade=1, "non base" means min_grade=1.
- technique_categories_include: dish must use at least one technique from these categories
  (references to "Manuale di Cucina di Sirius Cosmo")
- distance_constraint: restaurant must be within max_ly light-years of origin planet

IMPORTANT RULES:
1. Use EXACT names from the reference lists below — do not paraphrase or translate
2. Set unused fields to null (not empty arrays)
3. "esclusivamente X" (exclusively X) with "evitando Y" means: techniques_include=[X], techniques_exclude=[Y]
4. "senza" / "evitando" / "non" / "escludendo" = exclude
5. "o" between items = OR (use *_any fields), "e" between items = AND (use *_include fields)
6. For technique category references like "tecnica di taglio del Manuale", use technique_categories_include
7. Return ONLY valid JSON, no markdown fences, no commentary

KNOWN INGREDIENTS:
{ingredients}

KNOWN TECHNIQUES:
{techniques}

KNOWN RESTAURANTS:
{restaurants}

KNOWN PLANETS:
{planets}

TECHNIQUE CATEGORIES (from Manuale di Cucina):
{technique_categories}

Question: {question}
"""


class QueryParser:
    """Parse questions into structured filters using LLM."""

    def __init__(self, dishes_df: pd.DataFrame):
        self.client = OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.0,
        )
        self.ingredients = sorted(
            {ing for ings in dishes_df["ingredients"] for ing in ings}
        )
        self.techniques = sorted(
            {tech for techs in dishes_df["techniques"] for tech in techs}
        )
        self.restaurants = sorted(dishes_df["restaurant"].unique())
        self.planets = sorted(dishes_df["planet"].unique())
        self.technique_categories = load_technique_categories()

    def _build_prompt(self, question: str) -> str:
        categories_str = "\n".join(
            f"- {cat}: {', '.join(techs)}"
            for cat, techs in self.technique_categories.items()
        )
        return _PARSER_PROMPT_TEMPLATE.format(
            ingredients="\n".join(f"- {i}" for i in self.ingredients),
            techniques="\n".join(f"- {t}" for t in self.techniques),
            restaurants="\n".join(f"- {r}" for r in self.restaurants),
            planets="\n".join(f"- {p}" for p in self.planets),
            technique_categories=categories_str,
            question=question,
        )

    def parse(self, question: str) -> dict:
        """Parse a question into a structured filter dict."""
        prompt = self._build_prompt(question)
        response = self.client.invoke(prompt)

        return parse_json_response(response.text)
