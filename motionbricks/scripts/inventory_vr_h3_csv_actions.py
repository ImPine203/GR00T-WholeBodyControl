"""Inventory VR H3 CSV action names after stripping trial/actor suffixes."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


SUFFIX_PATTERNS = [
    re.compile(r"_\d+__A\d+(?:_M)?$"),
    re.compile(r"_\d+_A\d+(?:_M)?$"),
    re.compile(r"__A\d+(?:_M)?$"),
]


def normalize_action_name(stem: str):
    action = stem
    for pattern in SUFFIX_PATTERNS:
        action = pattern.sub("", action)
    return action


def write_tsv(path: Path, header: list[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(item) for item in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inventory VR H3 CSV action names")
    parser.add_argument("--input_dir", type=Path, default=Path("datas/vr_h3_data/vr_h3_1"))
    parser.add_argument("--output_dir", type=Path, default=Path("out/vr_h3_data_inventory"))
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    paths = sorted(args.input_dir.rglob("*.csv"))
    action_counts = Counter()
    family_counts = Counter()
    examples = defaultdict(list)

    for path in paths:
        action = normalize_action_name(path.stem)
        family = action.split("_")[0] if action else ""
        action_counts[action] += 1
        family_counts[family] += 1
        if len(examples[action]) < args.examples:
            examples[action].append(str(path.relative_to(args.input_dir)))

    action_rows = [
        (action, count, " | ".join(examples[action]))
        for action, count in action_counts.most_common()
    ]
    family_rows = family_counts.most_common()
    walk_rows = [
        (action, count, " | ".join(examples[action]))
        for action, count in sorted(action_counts.items())
        if "walk" in action.lower()
    ]
    candidate_walk_rows = [
        (action, count, " | ".join(examples[action]))
        for action, count in sorted(action_counts.items())
        if (
            ("walk_forward" in action.lower() or "walk_ff_loop" in action.lower() or "walk_180" in action.lower())
            and not any(token in action.lower() for token in [
                "injured", "zombie", "crouch", "stealth", "dance", "combat", "gun",
                "box", "turn", "start", "stop", "sideway", "dog", "hands_on_back",
                "talk", "watering", "pull", "carry", "drink", "sit",
            ])
        )
    ]

    write_tsv(args.output_dir / "actions.tsv", ["action", "count", "examples"], action_rows)
    write_tsv(args.output_dir / "families.tsv", ["family", "count"], family_rows)
    write_tsv(args.output_dir / "walk_actions.tsv", ["action", "count", "examples"], walk_rows)
    write_tsv(args.output_dir / "candidate_forward_walk.tsv", ["action", "count", "examples"], candidate_walk_rows)

    print(f"CSV files: {len(paths)}")
    print(f"Unique normalized actions: {len(action_counts)}")
    print(f"Output dir: {args.output_dir}")
    print("Top families:")
    for family, count in family_rows[:30]:
        print(f"  {family}: {count}")
    print("Candidate forward-walk actions:")
    for action, count, example in candidate_walk_rows[:80]:
        print(f"  {action}: {count} ({example})")


if __name__ == "__main__":
    main()
