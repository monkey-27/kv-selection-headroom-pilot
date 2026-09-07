"""Modal orchestration for cost-probed, resumable KV-headroom execution."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import modal


APP_NAME = "kv-selection-headroom-pilot"
OUTPUT_VOLUME_NAME = "kv-selection-headroom-output"
CACHE_VOLUME_NAME = "kv-selection-headroom-hf-cache"

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.1",
        "transformers==5.0.0",
        "datasets>=3.0",
        "accelerate>=1.0",
        "matplotlib>=3.8",
        "numpy>=1.26",
        "sympy",
        "latex2sympy2",
        "regex",
        "word2number",
        "sentencepiece",
        "protobuf",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_CACHE": "/cache/huggingface/hub",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "KVH_MODAL_VOLUME_NAME": OUTPUT_VOLUME_NAME,
        "PYTHONPATH": "/root",
    })
    .add_local_python_source("kvheadroom", "kvcompress")
    .add_local_file("configs/kv_headroom.json", "/root/configs/kv_headroom.json")
)

common = dict(
    image=image,
    secrets=[hf_secret],
    volumes={"/outputs": output_volume, "/cache": cache_volume},
    cpu=4,
    memory=32768,
)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _prepare_data() -> None:
    from datasets import load_dataset

    math_path = Path("/outputs/data/math/test.jsonl")
    gpqa_path = Path("/outputs/data/gpqa/test.jsonl")
    manifest_path = Path("/outputs/data/kv_headroom_source_manifest.json")
    if math_path.exists() and gpqa_path.exists() and manifest_path.exists():
        return
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    gpqa_raw = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    letters, gpqa = "ABCD", []
    for index, row in enumerate(gpqa_raw):
        correct = row["Correct Answer"]
        choices = [correct] + [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
        random.Random(20260906 + index).shuffle(choices)
        question = row["Question"] + "\n" + "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, choices)
        )
        gpqa.append({"question": question, "answer": letters[choices.index(correct)], "source_index": index})
    _write_jsonl(math_path, math)
    _write_jsonl(gpqa_path, gpqa)
    manifest_path.write_text(json.dumps({
        "math500": {"hub_id": "HuggingFaceH4/MATH-500", "split": "test",
                    "fingerprint": math._fingerprint, "rows": len(math)},
        "gpqa_diamond": {"hub_id": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                         "fingerprint": gpqa_raw._fingerprint, "rows": len(gpqa),
                         "choice_shuffle_seed": "20260906 + source_index"},
    }, indent=2) + "\n")
    output_volume.commit()


def _remote_config(gpu: str) -> str:
    source = json.loads(Path("/root/configs/kv_headroom.json").read_text())
    source["datasets"]["math500"]["path"] = "/outputs/data/math/test.jsonl"
    source["datasets"]["gpqa_diamond"]["path"] = "/outputs/data/gpqa/test.jsonl"
    source["paths"]["run_dir"] = "/outputs/kv_headroom_v1"
    # One physical GPU-hour is conservatively charged as one H100-equivalent
    # hour even if the selected device is an A100.
    source["runtime"]["gpu_h100_equivalent_factor"] = 1.0
    source["runtime"]["selected_gpu"] = gpu
    path = Path(f"/tmp/kv_headroom_{gpu.lower()}.json")
    path.write_text(json.dumps(source, indent=2) + "\n")
    return str(path)


def _benchmark(gpu: str) -> dict:
    import torch
    from kvheadroom.replay import QwenReplayScorer, ReplayLayout
    from kvheadroom.synthetic import build_examples

    cfg = json.loads(Path("/root/configs/kv_headroom.json").read_text())
    t0 = time.perf_counter()
    scorer = QwenReplayScorer.load(
        cfg["model"], cfg["dtype"], cfg["attention_backend"], cfg["model_revision"]
    )
    load_seconds = time.perf_counter() - t0
    trace = build_examples(scorer.tokenizer, 1)[0]
    layout = ReplayLayout.main(len(trace["prompt_ids"]))
    prefix = scorer.prefill(trace["prompt_ids"], trace["reasoning_ids"], layout)
    masks = [mask for mask in range(1 << 12) if mask.bit_count() == 6][:96]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for start in range(0, len(masks), 32):
        scorer.score_masks(prefix, trace["reasoning_ids"], layout, masks[start:start + 32])
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - t1
    result = {
        "gpu_request": gpu, "gpu_actual": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds, "subsets": len(masks), "score_seconds": score_seconds,
        "subsets_per_second": len(masks) / score_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch": torch.__version__,
    }
    Path(f"/outputs/benchmark_{gpu.lower()}.json").write_text(json.dumps(result, indent=2) + "\n")
    output_volume.commit(); cache_volume.commit()
    return result


def _run_full(gpu: str) -> dict:
    from kvheadroom.analyze import analyze
    from kvheadroom.exhaustive import run as exhaustive
    from kvheadroom.free_generation import run as free_generation
    from kvheadroom.traces import collect

    _prepare_data()
    config_path = _remote_config(gpu)
    collect(config_path)
    exhaustive("synthetic", config_path)
    exhaustive("main", config_path)
    analyze(config_path)
    free_generation(config_path)
    exhaustive("late", config_path)
    result = analyze(config_path)
    output_volume.commit(); cache_volume.commit()
    return result


@app.function(gpu="A100-40GB", timeout=3600, **common)
def benchmark_a100():
    return _benchmark("A100-40GB")


@app.function(gpu="H100", timeout=3600, **common)
def benchmark_h100():
    return _benchmark("H100")


@app.function(gpu="A100-40GB", timeout=36000, **common)
def run_full_a100():
    return _run_full("A100-40GB")


@app.function(gpu="H100", timeout=36000, **common)
def run_full_h100():
    return _run_full("H100")
