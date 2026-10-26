"""Eval harness for the dense baseline: validate ground truth, then measure precision@1."""
import contextlib
import csv
import sys

# search opens the local Qdrant storage at import; reuse its client (the folder allows one client per process).
from search import client, dense_search

REQUESTS_PATH = "data/labelled_missing_brick_requests.csv"
PARTS_PATH = "data/parts_75192_full.csv"
COLLECTION = "parts_75192"
REPORT_PATH = "outputs/baseline_failure_report.txt"

# Free-text typed colour -> the parts-list colour names it can mean. Bare "grey" covers both greys.
TYPED_COLOURS = {
    "grey": {"Light Bluish Gray", "Dark Bluish Gray"},
    "bluish grey": {"Light Bluish Gray", "Dark Bluish Gray"},
    "light grey": {"Light Bluish Gray"},
    "light gray": {"Light Bluish Gray"},
    "light bluish grey": {"Light Bluish Gray"},
    "light bluish gray": {"Light Bluish Gray"},
    "medium stone grey": {"Light Bluish Gray"},
    "lbg": {"Light Bluish Gray"},
    "dark grey": {"Dark Bluish Gray"},
    "dark bluish grey": {"Dark Bluish Gray"},
    "dark bluish gray": {"Dark Bluish Gray"},
    "dark stone grey": {"Dark Bluish Gray"},
    "black": {"Black"},
    "white": {"White"},
    "tan": {"Tan"},
    "blue": {"Blue"},
    "clear": {"Trans-Clear"},
    "transparent": {"Trans-Clear"},
    "trans clear": {"Trans-Clear"},
    "transparent blue": {"Trans-Light Blue"},
    "trans light blue": {"Trans-Light Blue"},
}

# Tie-break order: the first rule that applies to a MISS wins.
FAILURE_KINDS = [
    ("conflict", "typed part number + typed colour, and no parts-list element has that design in that colour"),
    ("wrong_colour", "top-1 has the ground-truth design number but a different colour"),
    ("lookalike_design", "top-1 has a different design number from the ground truth"),
    ("duplicate_element", "top-1 has the ground-truth design and colour but another element_id"),
]


class UnclassifiedMiss(Exception):
    """A miss that matches none of the failure kinds; the rule needs a new kind, not a guess."""


def load_requests(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_parts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_conflict(request, parts):
    """True when the typed part number and typed colour cannot both be right for any element.

    A typed number that is an element_id is checked against that element's own colour;
    otherwise it is treated as a design number. Numbers that match neither are conflicts.
    """
    number = request["typed_part_number"].strip()
    typed_colour = request["typed_colour"].strip().lower()
    if not number or not typed_colour:
        return False
    colours = TYPED_COLOURS[typed_colour]  # KeyError on an unmapped colour, so it gets added rather than guessed.
    by_element = [p for p in parts if p["element_id"] == number]
    candidates = by_element or [p for p in parts if p["design_number"] == number]
    return not any(p["colour_name"] in colours for p in candidates)


def failure_kind(request, top1_design, top1_colour, parts):
    if is_conflict(request, parts):
        return "conflict"
    truth_design = request["ground_truth_design_number"].strip()
    truth_colour = request["ground_truth_colour_name"].strip()
    if top1_design == truth_design and top1_colour != truth_colour:
        return "wrong_colour"
    if top1_design != truth_design:
        return "lookalike_design"
    if top1_colour == truth_colour:
        return "duplicate_element"
    raise UnclassifiedMiss(
        f"{request['request_id']}: top-1 matches the ground-truth design {truth_design} and colour "
        f"{truth_colour} but no failure kind covers this"
    )


def load_collection_payloads(client):
    """Map element_id -> payload for every point in the collection."""
    payloads = {}
    offset = None
    while True:
        points, offset = client.scroll(COLLECTION, limit=1000, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            payloads[p.payload["element_id"]] = p.payload
        if offset is None:
            return payloads


def check_ground_truth(requests, payloads):
    """Return (distinct ground truths, missing IDs, payload mismatches)."""
    truths = {}
    for r in requests:
        eid = r["ground_truth_element_id"].strip()
        truths.setdefault(eid, []).append(r)

    missing = sorted(eid for eid in truths if eid not in payloads)

    mismatches = []
    for eid, rows in sorted(truths.items()):
        if eid not in payloads:
            continue
        stored = payloads[eid]
        for r in rows:
            design = r["ground_truth_design_number"].strip()
            colour = r["ground_truth_colour_name"].strip()
            if design != stored["design_number"] or colour != stored["colour_name"]:
                mismatches.append((r["request_id"], eid, design, colour, stored["design_number"], stored["colour_name"]))
    return truths, missing, mismatches


def precision_at_1(requests, parts):
    """Return (hits, rows) where rows are (request_id, top-1 element_id, ground truth, hit, failure kind)."""
    rows = []
    for r in requests:
        top1, top1_design, top1_colour = dense_search(r["request_text"], k=1)[0]
        truth = r["ground_truth_element_id"].strip()
        hit = top1 == truth
        kind = "" if hit else failure_kind(r, top1_design, top1_colour, parts)
        rows.append((r["request_id"], top1, truth, hit, kind))
    return sum(row[3] for row in rows), rows


def conflict_detection(requests, parts):
    """Compare is_conflict against the is_conflicting label on every request, hit or miss.

    Returns (confusion counts, false negatives, false positives, per-pattern [labelled, caught]).
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    false_negatives, false_positives = [], []
    patterns = {}
    for r in requests:
        predicted = is_conflict(r, parts)
        labelled = r["is_conflicting"].strip().lower() == "true"
        if labelled:
            stats = patterns.setdefault(r["conflict_pattern"].strip(), [0, 0])
            stats[0] += 1
            stats[1] += predicted
        if predicted and labelled:
            counts["tp"] += 1
        elif predicted:
            counts["fp"] += 1
            false_positives.append(r)
        elif labelled:
            counts["fn"] += 1
            false_negatives.append(r)
        else:
            counts["tn"] += 1
    return counts, false_negatives, false_positives, patterns


def print_conflict_rows(title, rows):
    print(f"{title}: {len(rows)}")
    if not rows:
        return
    print(f"  {'request_id':<10}  {'typed_part_number':<17}  {'typed_colour':<18}  conflict_pattern")
    for r in rows:
        print(f"  {r['request_id']:<10}  {r['typed_part_number']:<17}  {r['typed_colour']:<18}  {r['conflict_pattern']}")


class Tee:
    """Write to several streams at once, so the saved report is exactly what was printed."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def print_report(requests, payloads, parts):
    truths, missing, mismatches = check_ground_truth(requests, payloads)

    print("Dense baseline failure report")
    print()
    print("1) Ground-truth check")
    print(f"Requests loaded: {len(requests)}")
    print(f"Points in {COLLECTION}: {len(payloads)}")
    print(f"Distinct ground-truth element_ids: {len(truths)}")
    print(f"Found in collection: {len(truths) - len(missing)} / {len(truths)}")
    if missing:
        print("Missing from collection:")
        for eid in missing:
            req_ids = ", ".join(r["request_id"] for r in truths[eid])
            print(f"  {eid}  (requests: {req_ids})")
    else:
        print("Missing from collection: none")
    if mismatches:
        print(f"Design/colour mismatches vs Qdrant payload: {len(mismatches)}")
        for req_id, eid, design, colour, s_design, s_colour in mismatches:
            print(f"  {req_id} element_id={eid}: labelled {design} / {colour}  |  stored {s_design} / {s_colour}")
    else:
        print("Design/colour mismatches vs Qdrant payload: none")

    # A failed ground-truth check means any score would stand on rows we can't trust,
    # so stop here and say why instead of scoring what's left.
    if missing or mismatches:
        n_missing_reqs = sum(len(truths[eid]) for eid in missing)
        print()
        print("STOPPED: ground truth does not resolve, so nothing is scored.")
        if missing:
            print(f"  {len(missing)} answer IDs ({n_missing_reqs} requests) are not in {COLLECTION}: the collection")
            print("  was built from the 80-row parts list and these elements were never ingested.")
        if mismatches:
            print(f"  {len(mismatches)} requests contradict the stored design/colour (e.g. 4211387/4211388 greys")
            print("  swapped, 3069b/3070b and 3022/3023 crossed); either the labels or the parts list is wrong.")
        print("  Fix: ingest the full 75192 inventory and reconcile the contradicting labels, then rerun.")
        return False

    hits, rows = precision_at_1(requests, parts)
    counts, false_negatives, false_positives, patterns = conflict_detection(requests, parts)
    hit_by_request = {row[0]: row[3] for row in rows}

    print()
    print("2) Dense precision@1")
    print(f"Dense precision@1: {hits}/{len(rows)} = {hits / len(rows):.1%}  ({len(rows)} of {len(requests)} requests scored)")

    print()
    print("3) Failure kinds (tie-break order, first match wins):")
    for i, (kind, rule) in enumerate(FAILURE_KINDS, 1):
        print(f"  {i}) {kind}: {rule}")
    for kind, _ in FAILURE_KINDS:
        print(f"  {kind:<17} {sum(row[4] == kind for row in rows)}")
    print(f"  {'total misses':<17} {len(rows) - hits}")
    print()
    print("  Misses, one row each, grouped by failure kind:")
    print(f"  {'failure_kind':<17}  {'request_id':<10}  {'top1_element_id':<15}  ground_truth")
    for kind, _ in FAILURE_KINDS:
        for req_id, top1, truth, hit, row_kind in rows:
            if not hit and row_kind == kind:
                print(f"  {kind:<17}  {req_id:<10}  {top1:<15}  {truth}")

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    print()
    print(f"4) Conflict detection vs is_conflicting (all {len(requests)} requests):")
    print(f"  {'':<18}  {'labelled true':>13}  {'labelled false':>14}")
    print(f"  {'predicted true':<18}  {f'TP {tp}':>13}  {f'FP {fp}':>14}")
    print(f"  {'predicted false':<18}  {f'FN {fn}':>13}  {f'TN {tn}':>14}")
    precision = f"{tp}/{tp + fp} = {tp / (tp + fp):.1%}" if tp + fp else "n/a (no predicted conflicts)"
    recall = f"{tp}/{tp + fn} = {tp / (tp + fn):.1%}" if tp + fn else "n/a (no labelled conflicts)"
    print(f"  precision: {precision}")
    print(f"  recall:    {recall}")
    print()
    conflicting = [r for r in requests if r["is_conflicting"].strip().lower() == "true"]
    print(f"Conflicting-request list (is_conflicting = true): {len(conflicting)}")
    print(f"  {'request_id':<10}  {'conflict_pattern':<28}  {'flagged_by_us':<13}  search")
    for r in conflicting:
        flagged = "yes" if is_conflict(r, parts) else "no"
        result = "HIT" if hit_by_request[r["request_id"]] else "MISS"
        print(f"  {r['request_id']:<10}  {r['conflict_pattern']:<28}  {flagged:<13}  {result}")
    print()
    print("Labelled conflicts by conflict_pattern:")
    print(f"  {'conflict_pattern':<28}  {'labelled':>8}  {'caught':>6}")
    for pattern, (labelled, caught) in sorted(patterns.items()):
        print(f"  {pattern or '(blank)':<28}  {labelled:>8}  {caught:>6}")
    print()
    print_conflict_rows("False negatives (labelled conflict, not flagged)", false_negatives)
    print_conflict_rows("False positives (flagged, not labelled conflict)", false_positives)

    print()
    print("5) Per-request results")
    print(f"{'request_id':<10}  {'top1_element_id':<15}  {'ground_truth':<12}  {'result':<6}  failure_kind")
    for req_id, top1, truth, hit, kind in rows:
        print(f"{req_id:<10}  {top1:<15}  {truth:<12}  {'HIT' if hit else 'MISS':<6}  {kind}")


def main():
    requests = load_requests(REQUESTS_PATH)
    payloads = load_collection_payloads(client)
    parts = load_parts(PARTS_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as report, contextlib.redirect_stdout(Tee(sys.stdout, report)):
        complete = print_report(requests, payloads, parts)
    if complete is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
