from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from .config import ROOT, freeze_config, load_config
from .io import append_jsonl, read_jsonl
from .runtime import GPUHourBudget


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    choices = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?", text, re.I)
    return choices[-1].upper() if choices else None


def _math_correct(text: str, row: dict) -> bool:
    # Reuse Random Attention's benchmark-grade symbolic equivalence.
    from kvcompress.harness.Utils.grader import check_is_correct
    from kvcompress.harness.Utils.parser import extract_answer, parse_ground_truth

    gold = parse_ground_truth(row, "math")[1]
    return bool(check_is_correct(extract_answer(text, "math"), gold))


def _gpqa_gold(row: dict) -> str:
    for key in ("answer", "correct", "correct_answer", "label"):
        if key in row and str(row[key]).strip():
            value = str(row[key]).strip()
            if value.upper() in "ABCD" and len(value) == 1:
                return value.upper()
    # VaSE-format GPQA is normalized by its parser.
    from kvcompress.harness.Utils.parser import parse_ground_truth
    return str(parse_ground_truth(row, "gpqa")[1]).strip().upper()


def _question(row: dict, dataset: str) -> str:
    from kvcompress.harness.Utils.parser import parse_question
    return parse_question(row, "math" if dataset == "math500" else "gpqa")


def _prompt(tokenizer, row: dict, dataset: str):
    question = _question(row, dataset)
    if dataset == "math500":
        content = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    else:
        content = question + "\nReason carefully. End with the letter of the best answer in \\boxed{}."
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True,
    )
    return list(encoded["input_ids"]), content


def _correct(text: str, row: dict, dataset: str) -> bool:
    if dataset == "math500":
        return _math_correct(text, row)
    return _boxed(text) == _gpqa_gold(row)


def collect(config_path: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(config_path)
    run_dir = Path(cfg["paths"]["run_dir"])
    freeze_config(cfg)
    out = run_dir / "traces" / "selected.jsonl"
    candidates_out = run_dir / "traces" / "candidates.jsonl"
    selected = read_jsonl(out)
    selected_keys = {(r["dataset"], r["example_index"]) for r in selected}
    attempted = {(r["dataset"], r["example_index"]) for r in read_jsonl(candidates_out)}
    rt = cfg["runtime"]
    budget = GPUHourBudget(run_dir, rt["max_h100_equivalent_hours"],
                           rt["gpu_h100_equivalent_factor"], rt["stop_margin_hours"])

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["model_revision"], torch_dtype=getattr(torch, cfg["dtype"]),
        device_map="cuda:0", attn_implementation=cfg["attention_backend"],
    ).eval()
    import transformers
    provenance = {
        "upstream_random_attention_commit": "64db9688a12f0926db45fec039370f6ceb1ab4fe",
        "model": cfg["model"], "requested_model_revision": cfg["model_revision"],
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("resolved_model_commit") != provenance["resolved_model_commit"]:
            raise RuntimeError("resolved model revision changed across resume")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    rng = random.Random(cfg["seed"])
    gen = cfg["generation"]
    required = cfg["main"]["checkpoint_reasoning_token"] + cfg["main"]["target_tokens"]

    with budget.active("collect_traces"):
        for dataset, dcfg in cfg["datasets"].items():
            rows = read_jsonl(_resolve(dcfg["path"]))
            if not rows:
                raise FileNotFoundError(
                    f"missing {dataset} data at {_resolve(dcfg['path'])}; run scripts/download_data.py"
                )
            order = list(range(min(len(rows), int(dcfg["max_candidates"]))))
            rng.shuffle(order)
            have = sum(r["dataset"] == dataset for r in selected)
            for index in order:
                if have >= int(dcfg["target_traces"]):
                    break
                if (dataset, index) in attempted or (dataset, index) in selected_keys:
                    continue
                budget.check()
                prompt_ids, prompt_text = _prompt(tok, rows[index], dataset)
                x = torch.tensor([prompt_ids], device=model.device)
                torch.manual_seed(cfg["seed"] + index + (10000 if dataset == "gpqa_diamond" else 0))
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=x,
                        max_new_tokens=gen["max_new_tokens"],
                        do_sample=gen["do_sample"], temperature=gen["temperature"],
                        top_p=gen["top_p"], use_cache=True,
                    )
                reasoning_ids = output[0, len(prompt_ids):].tolist()
                text = tok.decode(reasoning_ids, skip_special_tokens=True)
                candidate = {
                    "dataset": dataset, "example_index": index,
                    "reasoning_tokens": len(reasoning_ids),
                    "correct": _correct(text, rows[index], dataset),
                }
                append_jsonl(candidates_out, [candidate])
                attempted.add((dataset, index))
                if candidate["correct"] and len(reasoning_ids) >= required:
                    record = {
                        **candidate,
                        "trace_id": f"{dataset}-{index:04d}",
                        "prompt_ids": prompt_ids,
                        "reasoning_ids": reasoning_ids,
                        "prompt_text": prompt_text,
                        "completion_text": text,
                        "gold": rows[index],
                    }
                    append_jsonl(out, [record])
                    selected.append(record)
                    selected_keys.add((dataset, index))
                    have += 1
                budget.flush(f"collect_{dataset}_{index}")
            if have < int(dcfg["target_traces"]):
                raise RuntimeError(f"only collected {have}/{dcfg['target_traces']} qualifying {dataset} traces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "kv_headroom.json"))
    args = parser.parse_args()
    collect(args.config)


if __name__ == "__main__":
    main()
