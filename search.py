"""Dense (meaning-only) baseline search over the parts_75192 Qdrant collection."""
import sys

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

STORAGE_PATH = "qdrant_storage"
COLLECTION = "parts_75192"
MODEL = "BAAI/bge-small-en-v1.5"

# Stand-in for Leo's missing piece: Plate 1 x 2, Dark Bluish Gray, design 3023.
TARGET_ELEMENT_ID = "4211091"
TARGET_COLOUR = "Dark Bluish Gray"

model = TextEmbedding(MODEL)
client = QdrantClient(path=STORAGE_PATH)


def dense_search(query, k=5):
    """Return the top-k hits, ranked, as (element_id, design_number, colour_name)."""
    vector = next(iter(model.embed([query]))).tolist()
    hits = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True).points
    return [(h.payload["element_id"], h.payload["design_number"], h.payload["colour_name"]) for h in hits]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "set 75192 dark grey piece 3023"
    print(f"Query: {query!r}")
    results = dense_search(query)
    for rank, (eid, design, colour) in enumerate(results, 1):
        print(f"{rank}. element_id={eid}  design_number={design}  colour_name={colour}")

    top_colour = results[0][2]
    verdict = "RIGHT colour" if top_colour == TARGET_COLOUR else "WRONG colour"
    print(f"Verdict: top result is {top_colour} -> {verdict} (want {TARGET_COLOUR})")


if __name__ == "__main__":
    main()
