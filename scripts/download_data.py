#!/usr/bin/env python3
"""Materialize the two frozen benchmark sources in Random Attention JSONL layout."""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main():
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    gpqa_raw = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    gpqa = []
    letters = "ABCD"
    for index, row in enumerate(gpqa_raw):
        correct = row["Correct Answer"]
        choices = [correct] + [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
        random.Random(20260906 + index).shuffle(choices)
        answer = letters[choices.index(correct)]
        question = row["Question"] + "\n" + "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
        gpqa.append({"question": question, "answer": answer, "source_index": index})
    write(ROOT / "data" / "math" / "test.jsonl", math)
    write(ROOT / "data" / "gpqa" / "test.jsonl", gpqa)
    manifest = {
        "math500": {"hub_id": "HuggingFaceH4/MATH-500", "config": None, "split": "test",
                    "fingerprint": math._fingerprint, "rows": len(math)},
        "gpqa_diamond": {"hub_id": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                         "fingerprint": gpqa_raw._fingerprint, "rows": len(gpqa),
                         "choice_shuffle_seed": "20260906 + source_index"},
    }
    (ROOT / "data" / "kv_headroom_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote MATH-500={len(math)} GPQA-Diamond={len(gpqa)}")


if __name__ == "__main__":
    main()
