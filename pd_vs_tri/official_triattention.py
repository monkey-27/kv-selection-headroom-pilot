from __future__ import annotations
import hashlib, json, os, subprocess
from pathlib import Path

OFFICIAL_SHA="325297218a0d85cc83bc9ca1ecfa1a33a178831f"
TRTLLM_SHA="8a26dd8f9d6fd09781d7e6f5f1162674f9f8fd05"

def require_checkout(path:Path,sha:str):
    got=subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()
    if got!=sha:raise RuntimeError(f"official checkout mismatch: {path}: {got} != {sha}")

def calibration_hash(path:Path):
    if not path.is_file() or path.stat().st_size==0:raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def calibration_command(repo:Path,model_path:Path,input_path:Path,output_path:Path):
    require_checkout(repo,OFFICIAL_SHA)
    return [os.environ.get("PYTHON","python3"),str(repo/"scripts/calibrate.py"),"--model",str(model_path),
      "--input",str(input_path),"--output",str(output_path),"--max-length","32768","--device","cuda"]

def backend_for(hardware):
    cc=tuple(hardware.get("compute_capability",[]))
    return "tensorrt_llm" if cc in ((10,0),(10,3)) else "triattention_official_vllm"

def fail_closed_support(model,backend,smoke_path:Path):
    if not smoke_path.exists():raise RuntimeError(f"{model} {backend}: missing successful official 8-example smoke artifact")
    smoke=json.loads(smoke_path.read_text())
    if not smoke.get("passed") or smoke.get("model")!=model or smoke.get("backend")!=backend:raise RuntimeError(f"unsupported/unvalidated cell: {smoke_path}")
    return smoke
