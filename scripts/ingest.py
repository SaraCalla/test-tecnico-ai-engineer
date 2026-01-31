from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.type import DenseEmbedding
from datapizza.vectorstores.qdrant import QdrantVectorstore

from src.chunking import menus_to_chunks
from src.config import settings
from src.parsing import parse_all_menus


def embed_chunks(chunks):
    """Embed all chunks in batches using OpenAI, attaching vectors in-place."""
    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )

    texts = [c.text for c in chunks]
    for i in range(0, len(texts), settings.embedding_batch_size):
        batch_texts = texts[i : i + settings.embedding_batch_size]
        batch_vectors = embedder.embed(batch_texts)
        for j, vector in enumerate(batch_vectors):
            chunks[i + j].embeddings = [
                DenseEmbedding(name=settings.embedding_name, vector=vector)
            ]
        end = min(i + settings.embedding_batch_size, len(texts))
        print(f"  Embedded {end}/{len(texts)} chunks")


def store_chunks(chunks):
    """Create Qdrant collection and store embedded chunks."""
    vectorstore = QdrantVectorstore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    vector_config = VectorConfig(
        name=settings.embedding_name,
        dimensions=settings.embedding_dimensions,
        distance=Distance.COSINE,
    )
    vectorstore.create_collection(settings.qdrant_collection_name, [vector_config])
    vectorstore.add(chunks, collection_name=settings.qdrant_collection_name)

    print(f"  Stored {len(chunks)} chunks in '{settings.qdrant_collection_name}'")


def main():
    print("=== Step 1: Parsing menus ===")
    menus = parse_all_menus()
    total_dishes = sum(len(m.dishes) for m in menus)
    print(f"  {len(menus)} menus, {total_dishes} dishes\n")

    print("=== Step 2: Creating chunks ===")
    chunks = menus_to_chunks(menus)
    print(f"  {len(chunks)} chunks created\n")

    print("=== Step 3: Embedding chunks ===")
    embed_chunks(chunks)
    print()

    print("=== Step 4: Storing in Qdrant ===") # Docker is used to run Qdrant
    store_chunks(chunks)

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
