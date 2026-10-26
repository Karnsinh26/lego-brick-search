"""Dense (meaning-only) baseline search over the parts_75192 Qdrant collection."""
import sys

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

STORAGE_PATH = "qdrant_storage"
COLLECTION = "parts_75192"
MODEL = "BAAI/bge-small-en-v1.5"

# Stand-in for Leo's missing piece: Plate 1 x 2, Dark Bluish Gray.
TARGET_ELEMENT_ID = "4211091"
TARGET_COLOUR = "Dark Bluish Gray"

model = TextEmbedding(MODEL)
client = QdrantClient(path=STORAGE_PATH)


def dense_search(query, k=5):
    vector = next(iter(model.embed([query]))).tolist()
    hits = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True).points
    return [(h.payload["element_id"], h.payload["design_number"], h.payload["colour_name"], h.score) for h in hits]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "set 75192 dark grey piece 3023"
    print(f"Query: {query!r}")
    print(f"Target: {TARGET_ELEMENT_ID} ({TARGET_COLOUR}, design 3023)")
    results = dense_search(query)
    for rank, (eid, design, colour, score) in enumerate(results, 1):
        print(f"{rank}. {eid}  {design:<6} {colour:<18} score={score:.4f}")

    top_eid, _, top_colour, _ = results[0]
    print(f"Top result right colour ({TARGET_COLOUR})? {'YES' if top_colour == TARGET_COLOUR else 'NO'}")
    print(f"Top result exact target part? {'YES' if top_eid == TARGET_ELEMENT_ID else 'NO'}")
    found = [eid for eid, *_ in results]
    rank = found.index(TARGET_ELEMENT_ID) + 1 if TARGET_ELEMENT_ID in found else None
    print(f"Target rank in top 5: {rank if rank else 'not found'}")


if __name__ == "__main__":
    main()
