import re
import time

from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.rerankers.cohere import CohereReranker
from datapizza.type import Chunk
from datapizza.vectorstores.qdrant import QdrantVectorstore

from src.config import settings
from src.logger import logger
from src.parsing import load_dish_id_mapping

_ANSWER_PROMPT = """\
You are an assistant that answers questions about intergalactic restaurant dishes.

Below are dishes retrieved from a database. Use ONLY these dishes to answer the question.
Return ONLY the names of dishes that match the question, one per line.
If no dishes match, return "NONE".

Do NOT invent dishes. Do NOT include dishes that don't satisfy ALL the conditions in the question.

Retrieved dishes:
{context}

Question: {question}

Matching dish names (one per line):
"""


class RAGPipeline:
    """Retrieve-then-answer pipeline: Qdrant -> Cohere rerank -> OpenAI LLM."""

    def __init__(self):
        self.embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )
        self.vectorstore = QdrantVectorstore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.reranker = CohereReranker(
            api_key=settings.cohere_api_key,
            endpoint="https://api.cohere.com",
            model=settings.reranker_model,
            top_n=settings.reranker_top_n,
        )
        self.llm = OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        self.dish_name_to_id = load_dish_id_mapping()
        self._last_rerank_time = 0.0
        self._rerank_interval = 9  # seconds between rerank calls (< 10 req/min)

    def retrieve(self, question: str) -> list[Chunk]:
        """Embed question and search Qdrant for top-k similar chunks."""
        query_vector = self.embedder.embed(question)
        return self.vectorstore.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            k=settings.retrieval_top_k,
            vector_name=settings.embedding_name,
        )

    def rerank(self, question: str, chunks: list[Chunk], max_retries: int = 3) -> list[Chunk]:
        """Rerank retrieved chunks using Cohere, with proactive throttling and retry."""
        # Proactive throttle: wait enough time since last call to stay under 10 req/min
        now = time.time()
        elapsed = now - self._last_rerank_time
        if elapsed < self._rerank_interval:
            time.sleep(self._rerank_interval - elapsed)
        self._last_rerank_time = time.time()

        for attempt in range(max_retries):
            try:
                return self.reranker.rerank(query=question, documents=chunks)
            except Exception as e:
                if "429" in str(e) or "TooManyRequests" in type(e).__name__:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return self.reranker.rerank(query=question, documents=chunks)

    def answer(self, question: str, chunks: list[Chunk]) -> list[str]:
        """Send reranked context + question to LLM, return list of dish names."""
        context = "\n\n".join(chunk.text for chunk in chunks)
        prompt = _ANSWER_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        raw_text = response.text
        names = []
        for line in raw_text.strip().splitlines():
            line = line.strip()
            # Strip common LLM formatting: "1. ", "- ", "* ", "• "
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if line and line.upper() != "NONE":
                names.append(line)
        return names

    def names_to_ids(self, dish_names: list[str]) -> list[int]:
        """Map dish names from LLM response to dish IDs, logging mismatches."""
        ids = []
        for name in dish_names:
            dish_id = self.dish_name_to_id.get(name)
            if dish_id is not None:
                ids.append(dish_id)
            else:
                logger.warning(f"    LLM returned unknown dish: '{name}'")
        return ids

    def query(self, question: str) -> list[int]:
        """Full pipeline: question -> retrieve -> rerank -> answer -> dish IDs."""
        chunks = self.retrieve(question)
        logger.debug(f"       retrieved: {[c.metadata.get('dish_id') for c in chunks]}")
        reranked = self.rerank(question, chunks)
        logger.debug(f"       reranked: {[c.metadata.get('dish_id') for c in reranked]}")
        dish_names = self.answer(question, reranked)
        ids = self.names_to_ids(dish_names)
        logger.debug(f"       llm answer: {ids}")
        return ids
