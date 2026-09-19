from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import MODELS,atomic_json

def resolve(out,download_root=None):
    if out.exists():raise FileExistsError(out)
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi(); locked={}
    for model in MODELS:
        info=api.model_info(model,files_metadata=False); sha=info.sha
        row={"model":model,"revision":sha}
        if download_root:
            row["local_path"]=snapshot_download(model,revision=sha,cache_dir=download_root)
        locked[model]=row
    atomic_json(out,{"schema":"model-revisions-lock-v1","models":locked}); return locked

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--download-root",type=Path)
    a=p.parse_args(); print(json.dumps(resolve(a.output,a.download_root),indent=2))
if __name__=="__main__":main()

