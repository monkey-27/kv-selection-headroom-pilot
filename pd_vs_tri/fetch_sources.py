from __future__ import annotations
import argparse,hashlib,json,subprocess,urllib.request
from pathlib import Path
from .config import atomic_json,sha256

def fetch(lock_path,out):
    lock=json.loads(lock_path.read_text()); out.mkdir(parents=True,exist_ok=True); receipt={"schema":"dataset-fetch-receipt-v1","sources":{}}
    for name,spec in lock["sources"].items():
        revision=spec.get("revision")
        if not revision:raise RuntimeError(f"{name}: source revision must be frozen before download")
        target=out/name
        if spec.get("repository"):
            if not (target/".git").exists():subprocess.run(["git","clone",spec["repository"],str(target)],check=True)
            subprocess.run(["git","-C",str(target),"checkout","--detach",revision],check=True)
            got=subprocess.run(["git","-C",str(target),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()
            if got!=revision:raise RuntimeError(f"{name}: {got} != {revision}")
            receipt["sources"][name]={"kind":"git","revision":got,"path":str(target)}
        elif spec.get("canonical_url"):
            target.mkdir(parents=True,exist_ok=True); path=target/Path(spec["canonical_url"]).name
            if not path.exists():urllib.request.urlretrieve(spec["canonical_url"],path)
            receipt["sources"][name]={"kind":"url","url":spec["canonical_url"],"revision":revision,"sha256":sha256(path),"path":str(path)}
        elif spec.get("canonical_source"):
            raise RuntimeError(f"{name}: connector-specific canonical source requires an explicit exporter")
        else:raise RuntimeError(f"{name}: canonical source unresolved; refusing substitution")
    atomic_json(out/"fetch_receipt.json",receipt); return receipt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--lock",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(fetch(a.lock,a.output),indent=2))
if __name__=="__main__":main()
