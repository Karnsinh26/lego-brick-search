"""Load every distinct part of set 75192 into a local on-disk Qdrant collection."""
import csv
import uuid

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CSV_PATH = "data/rebrickable_75192_parts_list.csv"
STORAGE_PATH = "qdrant_storage"
COLLECTION = "parts_75192"
MODEL = "BAAI/bge-small-en-v1.5"


def load_parts(path):
    """One row per distinct element_id; spare rows never add duplicates."""
    parts = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["element_id"].strip()
            is_spare = row["is_spare"].strip().lower() == "true"
            # Keep the first row per element, but let a non-spare row replace a spare one.
            if eid not in parts or (not is_spare and parts[eid]["is_spare"].strip().lower() == "true"):
                parts[eid] = row
    return parts


def main():
    parts = load_parts(CSV_PATH)
    element_ids = list(parts)
    texts = [
        f"{parts[e]['design_number']} {parts[e]['colour_name']} {parts[e]['part_name']}"
        for e in element_ids
    ]

    model = TextEmbedding(MODEL)
    vectors = list(model.embed(texts))
    print(f"Embedded {len(vectors)} part descriptions")

    client = QdrantClient(path=STORAGE_PATH)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, eid)),
            vector=vec.tolist(),
            payload={
                "element_id": eid,
                "design_number": parts[eid]["design_number"],
                "colour_name": parts[eid]["colour_name"],
            },
        )
        for eid, vec in zip(element_ids, vectors)
    ]
    client.upsert(COLLECTION, points=points)

    point_count = client.count(COLLECTION, exact=True).count
    print(f"Qdrant points:        {point_count}")
    print(f"Distinct element_ids: {len(element_ids)}")
    print("Counts match" if point_count == len(element_ids) else "COUNTS DO NOT MATCH")


if __name__ == "__main__":
    main()
