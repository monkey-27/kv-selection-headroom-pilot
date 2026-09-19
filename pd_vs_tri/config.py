from __future__ import annotations
import hashlib, json
from pathlib import Path

MODELS = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-reasoning-plus",
)
DOMAINS = ("math", "constraint", "science", "code")
METHODS = ("pagedrop16", "triattention")

def load(path: Path) -> dict:
    data=json.loads(path.read_text())
    assert tuple(data["models"])==MODELS and tuple(data["domains"])==DOMAINS
    assert data["page_tokens"]==16 and data["recent_tokens"]==64
    assert data["bootstrap"]["replicates"]==10000 and data["bootstrap"]["seed"]==0
    return data

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def canonical_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n"); tmp.replace(path)

