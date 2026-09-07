from __future__ import annotations

from typing import Sequence


VALUES = (
    "cobalt", "saffron", "juniper", "marble", "velvet", "quartz", "indigo", "cedar",
    "topaz", "willow", "coral", "maple", "silver", "amber", "violet", "copper",
)


def _fit(tokenizer, text: str, length: int, filler: str = " Neutral ledger filler.") -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    fill = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) > length:
        raise ValueError(f"segment is {len(ids)} tokens, exceeds fixed length {length}")
    while len(ids) < length:
        ids.extend(fill[:length - len(ids)])
    return ids


def build_examples(tokenizer, n_examples: int = 16) -> list[dict]:
    if n_examples > len(VALUES):
        raise ValueError("not enough predeclared synthetic values")
    examples = []
    for i, value in enumerate(VALUES[:n_examples]):
        register = f"R{i:02d}"
        prompt_text = (
            "You are reading an immutable register ledger. A register is assigned exactly once. "
            "When asked later, return its assigned value exactly."
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=True,
            add_generation_prompt=True, return_dict=True,
        )["input_ids"]
        needed_block = (i * 7 + 3) % 12
        blocks = []
        for block in range(12):
            if block == needed_block:
                text = f" Ledger block {block}. Register {register} is assigned value {value}."
            else:
                text = f" Ledger block {block}. No register assignment occurs in this block."
            blocks.append(_fit(tokenizer, text, 64))
        recent = _fit(
            tokenizer,
            "The ledger is now closed. Preserve earlier assignments; do not invent or alter any value.",
            128,
        )
        query_text = f" Query: What exact value was assigned to register {register}? Answer: {value}."
        target = _fit(tokenizer, query_text, 64, filler=f" The answer remains {value}.")
        reasoning = [tok for block in blocks for tok in block] + recent + target
        value_ids = tokenizer.encode(value, add_special_tokens=False)
        context = reasoning[:896]
        occurrences = sum(context[j:j + len(value_ids)] == value_ids for j in range(len(context) - len(value_ids) + 1))
        if occurrences != 1:
            raise RuntimeError(f"synthetic {i}: value token sequence occurs {occurrences} times before target")
        query_ids = tokenizer.encode(
            f" Query: What exact value was assigned to register {register}? Answer:",
            add_special_tokens=False,
        )
        examples.append({
            "trace_id": f"synthetic-{i:02d}", "dataset": "synthetic",
            "prompt_ids": list(prompt_ids), "reasoning_ids": reasoning,
            "needed_block": needed_block, "needed_value": value,
            "query_ids": query_ids,
        })
    return examples
