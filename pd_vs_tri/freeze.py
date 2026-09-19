from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from .config import atomic_json,load,sha256
from .official_triattention import OFFICIAL_SHA,TRTLLM_SHA,calibration_hash

def freeze(root,template,out):
    if out.exists():raise FileExistsError(out)
    cfg=load(template); cfg["code_git_sha"]=subprocess.run(["git","rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()
    required=[root/"configs/model_revisions.lock.json",root/"configs/dataset_sources.lock.json",root/"configs/headline_budget_freeze.json",root/"configs/hardware.json"]
    for p in required:
        if not p.exists():raise FileNotFoundError(p)
    cfg["frozen_file_hashes"]={str(p.relative_to(root)):sha256(p) for p in required}
    model_lock=json.loads(required[0].read_text()); calibrations={}
    for model in cfg["models"]:
        safe=model.replace("/","__"); path=root/"calibration"/f"{safe}.pt"
        calibrations[model]={"path":str(path),"sha256":calibration_hash(path),"model_revision":model_lock["models"][model]["revision"]}
    cfg["calibrations"]=calibrations; cfg["official_triattention"]["sha"]=OFFICIAL_SHA; cfg["tensorrt_llm"]["sha"]=TRTLLM_SHA
    atomic_json(out,cfg); return cfg

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--template",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(freeze(a.root,a.template,a.output),indent=2))
if __name__=="__main__":main()
