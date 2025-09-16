import os
# Silence Hugging Face tokenizers fork/parallelism warning in dev reload scenarios
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import time
import json
from dotenv import load_dotenv
import asyncio

load_dotenv()

model = SentenceTransformer('minishlab/potion-retrieval-32M', device='cpu')

# Shared RAG state, initialized once at server startup
_collection_name = None
_qdrant_client = None

async def init_rag_system(collection_name):
    """Initialize the global RAG client and ensure the collection exists.

    This should be called once during server startup.
    """
    global _qdrant_client, _collection_name

    # Create async client
    _qdrant_client = AsyncQdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Ensure collection exists
    exists = await _qdrant_client.collection_exists(collection_name)
    if not exists:
        await _qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

    _collection_name = collection_name

    # Warm up the embedding model to avoid first-request latency
    try:
        # Run in a thread to avoid blocking the event loop
        await asyncio.to_thread(model.encode, ["warmup", "hello world"])
    except Exception as e:
        print(f"Embedding model warm-up failed: {e}")


async def shutdown_rag():
    """Clean up RAG resources on server shutdown."""
    global _qdrant_client
    try:
        if _qdrant_client and hasattr(_qdrant_client, "close"):
            await _qdrant_client.close()
    except Exception:
        # Best-effort cleanup; ignore if client doesn't support close
        pass

async def add_to_qdrant(scenarios):
    """Add scenarios to the shared RAG collection.

    Requires init_rag_system() to have been called.
    """
    if _qdrant_client is None or _collection_name is None:
        raise RuntimeError("RAG not initialized. Call init_rag_system() first.")
    points = []
    for index, scenario in enumerate(scenarios):
        json_text = json.dumps(scenario, sort_keys=True, ensure_ascii=False)
        # Offload encoding to a thread to avoid blocking the event loop
        vector = (await asyncio.to_thread(model.encode, json_text)).tolist()
        points.append(
            PointStruct(
                id=index,
                vector=vector,
                payload=scenario,
            )
        )
    if points:
        await _qdrant_client.upsert(collection_name=_collection_name, points=points)

async def rag_lookup(query):
    """Retrieve top results from the shared RAG collection for the given query."""
    if _qdrant_client is None or _collection_name is None:
        raise RuntimeError("RAG not initialized. Call init_rag_system() first.")

    # Offload encoding to a thread to avoid blocking the event loop
    embedding = await asyncio.to_thread(model.encode, query)

    results = await _qdrant_client.query_points(
        collection_name=_collection_name,
        query=embedding,
        limit=5,
        with_payload=True
    )
    points = results.points if hasattr(results, "points") else results

    def _one_line(text):
        if text is None:
            return ""
        return " ".join(str(text).splitlines())

    bullets_lines = []
    for p in points:
        payload = getattr(p, "payload", None) or {}
        raw_score = getattr(p, "score", None)
        if raw_score is None and isinstance(p, dict):
            raw_score = p.get("score")
        context = _one_line(payload.get("context"))
        guidelines = _one_line(payload.get("responseGuidelines"))
        if context or guidelines:
            bullets_lines.append(f"- {context}, {guidelines}")

    return "\n".join(bullets_lines)