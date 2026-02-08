import json

from src.config import settings
from src.dataframe import build_dishes_dataframe
from src.filter_engine import apply_filters
from src.logger import logger
from src.query_parser import QueryParser


class StructuredPipeline:
    """Question -> structured filter -> DataFrame lookup -> dish IDs."""

    def __init__(self):
        self.df = build_dishes_dataframe()
        self.parser = QueryParser(self.df)
        self.query_log: list[dict] = []
        logger.info(f"  Loaded {len(self.df)} dishes into DataFrame")

    def query(self, question: str) -> list[int]:
        """Parse question into filters and return matching dish IDs."""
        filters = self.parser.parse(question)
        logger.debug(f"       filters: {filters}")
        ids = apply_filters(self.df, filters)
        logger.debug(f"       matched: {ids}")
        self.query_log.append({
            "question": question,
            "filters": filters,
            "result_ids": ids,
        })
        return ids

    def save_log(self, path=None):
        """Save the query log to a JSON file for debugging."""
        if path is None:
            path = settings.output_dir / "structured_query_log.json"
        with open(path, "w") as f:
            json.dump(self.query_log, f, indent=2, ensure_ascii=False)
        logger.info(f"  Query log saved to: {path}")
