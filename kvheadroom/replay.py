from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .combinatorics import members


@dataclass(frozen=True)
class ReplayLayout:
    prompt_tokens: int
    checkpoint: int
    target_tokens: int
    recent_tokens: int
    blocks: tuple[tuple[int, ...], ...]  # reasoning-token indices, all before checkpoint-1

    def __post_init__(self):
        if self.checkpoint < 1 or self.target_tokens < 1:
            raise ValueError("checkpoint and target_tokens must be positive")
        flat = [i for block in self.blocks for i in block]
        if len(flat) != len(set(flat)):
            raise ValueError("blocks overlap")
        if any(i < 0 or i >= self.checkpoint - 1 for i in flat):
            raise ValueError("selectable blocks must precede the checkpoint query token")

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def prefix_reasoning_tokens(self) -> int:
        # The checkpoint's final token is replayed as the first query. This lets
        # pruning affect prediction of the first post-checkpoint token while the
        # query itself completes the protected recent-128 window.
        return self.checkpoint - 1

    def keep_indices(self, mask: int) -> list[int]:
        prompt = list(range(self.prompt_tokens))
        selected_reason = []
        for block_id in members(mask, self.n_blocks):
            selected_reason.extend(self.blocks[block_id])
        recent_lo = max(0, self.checkpoint - self.recent_tokens)
        # checkpoint-1 is the query token, so cache has the other recent-127.
        recent = list(range(recent_lo, self.checkpoint - 1))
        reasoning = sorted(set(selected_reason).union(recent))
        return prompt + [self.prompt_tokens + i for i in reasoning]

    @classmethod
    def main(cls, prompt_tokens: int, checkpoint: int = 896, target_tokens: int = 64,
             recent_tokens: int = 128, n_blocks: int = 12, block_tokens: int = 64):
        if n_blocks * block_tokens != checkpoint - recent_tokens:
            raise ValueError("main layout must exactly partition the 768 older reasoning tokens")
        blocks = tuple(tuple(range(b * block_tokens, (b + 1) * block_tokens)) for b in range(n_blocks))
        return cls(prompt_tokens, checkpoint, target_tokens, recent_tokens, blocks)

    @classmethod
    def late(cls, prompt_tokens: int, checkpoint: int, target_tokens: int = 64,
             recent_tokens: int = 128, n_blocks: int = 8):
        old = checkpoint - recent_tokens
        if old < n_blocks:
            raise ValueError("not enough old tokens for nonempty late-stress blocks")
        cuts = [round(i * old / n_blocks) for i in range(n_blocks + 1)]
        blocks = tuple(tuple(range(cuts[i], cuts[i + 1])) for i in range(n_blocks))
        return cls(prompt_tokens, checkpoint, target_tokens, recent_tokens, blocks)


def _filled_cache(prefix_cache, index_matrix: torch.Tensor):
    """Gather physical K/V slots; cached keys retain their original RoPE rotation."""
    from transformers import DynamicCache

    batch, kept = index_matrix.shape
    data = []
    for layer in prefix_cache.layers:
        keys, values = layer.keys, layer.values
        _, heads, _, dim = keys.shape
        idx = index_matrix[:, None, :, None].expand(batch, heads, kept, dim)
        out_k = torch.gather(keys.expand(batch, -1, -1, -1), 2, idx).contiguous()
        out_v = torch.gather(values.expand(batch, -1, -1, -1), 2, idx).contiguous()
        data.append((out_k, out_v))
    return DynamicCache(ddp_cache_data=data)


def _bottom_right_causal_mask(batch: int, query: int, past: int, dtype, device):
    """Additive mask for compacted physical caches with absolute-position RoPE."""
    minimum = torch.finfo(dtype).min
    mask = torch.full((query, past + query), minimum, dtype=dtype, device=device)
    mask[:, :past] = 0
    mask[:, past:] = torch.triu(torch.full((query, query), minimum, dtype=dtype, device=device), diagonal=1)
    return mask[None, None].expand(batch, 1, query, past + query)


class QwenReplayScorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, model_name: str, dtype: str = "bfloat16", attention_backend: str = "sdpa",
             revision: str = "main"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map="cuda:0",
            attn_implementation=attention_backend,
        ).eval()
        if model.config.model_type != "qwen3":
            raise ValueError(f"replay scorer is frozen to Qwen3, got {model.config.model_type}")
        return cls(model, tokenizer)

    @torch.inference_mode()
    def prefill(self, prompt_ids: Sequence[int], reasoning_ids: Sequence[int], layout: ReplayLayout):
        from transformers import DynamicCache

        ids = list(prompt_ids) + list(reasoning_ids[:layout.prefix_reasoning_tokens])
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        cache = DynamicCache(config=self.model.config)
        self.model(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=cache,
                   use_cache=True, logits_to_keep=1)
        expected = layout.prompt_tokens + layout.prefix_reasoning_tokens
        if cache.get_seq_length() != expected:
            raise RuntimeError(f"prefill cache length {cache.get_seq_length()} != {expected}")
        return cache

    def _inputs(self, reasoning_ids: Sequence[int], layout: ReplayLayout, batch: int):
        start = layout.checkpoint - 1
        stop = start + layout.target_tokens
        query = torch.tensor(reasoning_ids[start:stop], dtype=torch.long, device=self.device)
        labels = torch.tensor(reasoning_ids[start + 1:stop + 1], dtype=torch.long, device=self.device)
        if query.numel() != layout.target_tokens or labels.numel() != layout.target_tokens:
            raise ValueError(
                f"trace needs >= {layout.checkpoint + layout.target_tokens} reasoning tokens, "
                f"found {len(reasoning_ids)}"
            )
        positions = layout.prompt_tokens + torch.arange(start, stop, device=self.device)
        return query[None].expand(batch, -1), labels[None].expand(batch, -1), positions

    @torch.inference_mode()
    def score_masks(self, prefix_cache, reasoning_ids: Sequence[int], layout: ReplayLayout,
                    masks: Sequence[int]) -> list[dict]:
        keeps = [layout.keep_indices(mask) for mask in masks]
        lengths = {len(x) for x in keeps}
        if len(lengths) != 1:
            raise ValueError("batching across cardinalities is forbidden: gathered cache lengths differ")
        index_matrix = torch.tensor(keeps, dtype=torch.long, device=self.device)
        cache = _filled_cache(prefix_cache, index_matrix)
        batch, past = index_matrix.shape
        query, labels, positions = self._inputs(reasoning_ids, layout, batch)
        mask = _bottom_right_causal_mask(batch, layout.target_tokens, past,
                                         next(self.model.parameters()).dtype, self.device)
        output = self.model(
            input_ids=query,
            attention_mask={"full_attention": mask},
            position_ids=positions[None].expand(batch, -1),
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=layout.target_tokens,
        )
        logp = F.log_softmax(output.logits.float(), dim=-1)
        token_logp = logp.gather(-1, labels[..., None]).squeeze(-1)
        return [
            {
                "mask": int(subset),
                "k": int(subset).bit_count(),
                "utility": float(token_logp[i].mean().item()),
                "nll": float(-token_logp[i].mean().item()),
            }
            for i, subset in enumerate(masks)
        ]

    @torch.inference_mode()
    def score_untouched(self, prefix_cache, reasoning_ids: Sequence[int], layout: ReplayLayout) -> dict:
        """Score with the original, un-compacted full cache (the all-keep control)."""
        query, labels, positions = self._inputs(reasoning_ids, layout, 1)
        past = prefix_cache.get_seq_length()
        mask = _bottom_right_causal_mask(1, layout.target_tokens, past,
                                         next(self.model.parameters()).dtype, self.device)
        output = self.model(
            input_ids=query,
            attention_mask={"full_attention": mask},
            position_ids=positions[None], cache_position=positions,
            past_key_values=prefix_cache, use_cache=True,
            logits_to_keep=layout.target_tokens,
        )
        logp = F.log_softmax(output.logits.float(), dim=-1)
        token_logp = logp.gather(-1, labels[..., None]).squeeze(-1)
        utility = float(token_logp.mean().item())
        return {"utility": utility, "nll": -utility}

    @torch.inference_mode()
    def future_attention_mass(self, prefix_cache, reasoning_ids: Sequence[int], layout: ReplayLayout):
        """Mean full-cache attention mass over layers, query heads, and 64 future queries."""
        import transformers.models.qwen3.modeling_qwen3 as qwen3

        full_mask = (1 << layout.n_blocks) - 1
        keep = layout.keep_indices(full_mask)
        cache = _filled_cache(prefix_cache, torch.tensor([keep], device=self.device))
        query, _, positions = self._inputs(reasoning_ids, layout, 1)
        attn_mask = _bottom_right_causal_mask(1, layout.target_tokens, len(keep),
                                              next(self.model.parameters()).dtype, self.device)
        physical = {absolute: i for i, absolute in enumerate(keep)}
        block_slots = [
            torch.tensor([physical[layout.prompt_tokens + ridx] for ridx in block], device=self.device)
            for block in layout.blocks
        ]
        totals = torch.zeros(layout.n_blocks, dtype=torch.float64)
        calls = 0
        original = qwen3.eager_attention_forward

        def recording(module, query_states, key_states, value_states, attention_mask, scaling, **kwargs):
            nonlocal calls, totals
            output, weights = original(module, query_states, key_states, value_states,
                                       attention_mask, scaling, **kwargs)
            for b, slots in enumerate(block_slots):
                totals[b] += weights[..., slots].sum(-1).mean().double().cpu()
            calls += 1
            return output, weights

        old_backend = self.model.config._attn_implementation
        qwen3.eager_attention_forward = recording
        self.model.config._attn_implementation = "eager"
        try:
            self.model(
                input_ids=query,
                attention_mask={"full_attention": attn_mask},
                position_ids=positions[None],
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        finally:
            qwen3.eager_attention_forward = original
            self.model.config._attn_implementation = old_backend
        if calls != self.model.config.num_hidden_layers:
            raise RuntimeError(f"captured {calls} attention layers, expected {self.model.config.num_hidden_layers}")
        return (totals / calls).tolist()

    @torch.inference_mode()
    def generate_from_mask(self, prefix_cache, reasoning_ids: Sequence[int], layout: ReplayLayout,
                           subset_mask: int, max_new_tokens: int, temperature: float, top_p: float,
                           seed: int) -> list[int]:
        keep = layout.keep_indices(subset_mask)
        cache = _filled_cache(prefix_cache, torch.tensor([keep], device=self.device))
        current = torch.tensor([[reasoning_ids[layout.checkpoint - 1]]], device=self.device)
        position = layout.prompt_tokens + layout.checkpoint - 1
        generated: list[int] = []
        generator = torch.Generator(device=self.device).manual_seed(seed)
        eos = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            past = cache.get_seq_length()
            additive = torch.zeros((1, 1, 1, past + 1), dtype=next(self.model.parameters()).dtype,
                                   device=self.device)
            pos = torch.tensor([position], device=self.device)
            output = self.model(
                input_ids=current,
                attention_mask={"full_attention": additive},
                position_ids=pos[None], cache_position=pos,
                past_key_values=cache, use_cache=True, logits_to_keep=1,
            )
            logits = output.logits[0, -1].float()
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temperature, -1)
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                remove = torch.cumsum(sorted_p, -1) - sorted_p > top_p
                sorted_p[remove] = 0
                token = int(sorted_i[torch.multinomial(sorted_p, 1, generator=generator)])
            if token == eos:
                break
            generated.append(token)
            current = torch.tensor([[token]], device=self.device)
            position += 1
        return generated


def attention_topk_mask(masses: Sequence[float], k: int) -> int:
    order = sorted(range(len(masses)), key=lambda i: (-masses[i], i))
    return sum(1 << i for i in order[:k])
