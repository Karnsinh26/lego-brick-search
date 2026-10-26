# Dense search baseline · set 75192

Meaning-only search over every 75192 part. Later stages are measured against this.

- `ingest.py` loads each distinct element into a local on-disk Qdrant collection (`parts_75192`) using FastEmbed `BAAI/bge-small-en-v1.5`.
- `search.py` runs the dense search and prints the top 5 with a colour verdict.
- `outputs/` holds the saved terminal output of both runs.
