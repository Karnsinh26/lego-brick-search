# Dense search baseline · set 75192

Meaning-only search over every 75192 part. Later stages are measured against this.

- `ingest.py` loads each distinct element into a local on-disk Qdrant collection (`parts_75192`) using FastEmbed `BAAI/bge-small-en-v1.5`.
- `search.py` runs the dense search and prints the top 5 with a colour verdict.
- `outputs/` holds the saved terminal output of both runs.

## Data sources
- `data/rebrickable_75192_parts_list.csv`: the 80-row parts list from the Zero Stage 1 resources (used for Stage 1 only).
- `data/labelled_missing_brick_requests.csv`: the 100 hand-labelled requests from the Zero Stage 2 resources.
- `data/parts_75192_rebrickable.csv`: the current collection source, built from Rebrickable's public CSV dump (inventory 246263 of set 75192-1, 1069 elements). The raw dump goes in `data/rebrickable/` and is not committed.
- `eval.py` prints the rows where these sources disagree, beside each other, in `outputs/baseline_failure_report.txt`.
