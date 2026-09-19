from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import atomic_json
from .official_triattention import backend_for

def audit(model_lock,hardware,out):
    from transformers import AutoConfig
    backend=backend_for(hardware); cells={}
    for model,row in model_lock["models"].items():
        cfg=AutoConfig.from_pretrained(model,revision=row["revision"],trust_remote_code=True)
        model_type=cfg.model_type
        # Existing frozen attention patch is Qwen3-specific. Other PageDrop
        # architectures must remain blocked until a cache-equivalence smoke passes.
        page_status="candidate_existing_frozen" if model_type=="qwen3" else "requires_minimal_architecture_adapter_and_equivalence_smoke"
        cells[model]={"model_type":model_type,"pagedrop":page_status,"triattention_backend":backend,
          "triattention":"requires_official_calibration_and_8_item_smoke"}
    report={"schema":"pd-vs-tri-support-audit-v1","hardware":hardware,"cells":cells}
    atomic_json(out,report); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-lock",type=Path,required=True); p.add_argument("--hardware",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(audit(json.loads(a.model_lock.read_text()),json.loads(a.hardware.read_text()),a.output),indent=2))
if __name__=="__main__":main()

