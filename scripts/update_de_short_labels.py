"""
Update labels_a and labels_b in all de_short split files to match
the re-annotated labels in data/evaluation/gold_labels/full/gold_admin_de_short.jsonl.

Matching is done by (id, chunk_id).
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "data/evaluation/gold_labels/full/gold_admin_de_short.jsonl"
SPLITS = [
    ROOT / "data/evaluation/gold_labels/dev/gold_admin_de_short.jsonl",
    ROOT / "data/evaluation/gold_labels/dev/train/gold_admin_de_short.jsonl",
    ROOT / "data/evaluation/gold_labels/dev/val/gold_admin_de_short.jsonl",
    ROOT / "data/evaluation/gold_labels/test/gold_admin_de_short.jsonl",
]

# Build lookup: (id, chunk_id) -> {labels_a, labels_b}
gold = {}
with open(SOURCE) as f:
    for line in f:
        record = json.loads(line)
        key = (record["id"], record["chunk_id"])
        gold[key] = {
            "labels_a": record["labels_a"],
            "labels_b": record["labels_b"],
        }

print(f"Loaded {len(gold)} records from source.")

for split_path in SPLITS:
    updated, missing = 0, 0
    out_lines = []
    with open(split_path) as f:
        for line in f:
            record = json.loads(line)
            key = (record["id"], record["chunk_id"])
            if key in gold:
                record["labels_a"] = gold[key]["labels_a"]
                record["labels_b"] = gold[key]["labels_b"]
                updated += 1
            else:
                print(f"  WARNING: key {key} not found in source — left unchanged")
                missing += 1
            out_lines.append(json.dumps(record, ensure_ascii=False))

    with open(split_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"{split_path.relative_to(ROOT)}: {updated} updated, {missing} missing")
