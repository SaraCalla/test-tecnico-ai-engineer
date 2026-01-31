from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.rerankers.cohere import CohereReranker
from datapizza.type import Chunk
from datapizza.vectorstores.qdrant import QdrantVectorstore

from src.chunking import load_dish_id_mapping
from src.config import settings

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

    def retrieve(self, question: str) -> list[Chunk]:
        """Embed question and search Qdrant for top-k similar chunks."""
        query_vector = self.embedder.embed(question)
        return self.vectorstore.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            k=settings.retrieval_top_k,
            vector_name=settings.embedding_name,
        )

    def rerank(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        """Rerank retrieved chunks using Cohere."""
        return self.reranker.rerank(query=question, documents=chunks)

    def answer(self, question: str, chunks: list[Chunk]) -> list[str]:
        """Send reranked context + question to LLM, return list of dish names."""
        context = "\n\n".join(chunk.text for chunk in chunks)
        prompt = _ANSWER_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        raw_text = response.text
        lines = [line.strip() for line in raw_text.strip().splitlines()]
        return [line for line in lines if line and line != "NONE"]

    def names_to_ids(self, dish_names: list[str]) -> list[int]:
        """Map dish names from LLM response to dish IDs."""
        ids = []
        for name in dish_names:
            dish_id = self.dish_name_to_id.get(name)
            if dish_id is not None:
                ids.append(dish_id)
        return ids

    def query(self, question: str) -> list[int]:
        """Full pipeline: question -> retrieve -> rerank -> answer -> dish IDs."""
        chunks = self.retrieve(question)
        reranked = self.rerank(question, chunks)
        dish_names = self.answer(question, reranked)
        return self.names_to_ids(dish_names)
