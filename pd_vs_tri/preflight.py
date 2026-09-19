from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import sha256

def audit(root:Path,config:dict):
    checks=[]
    def check(name,path,required=True):
        ok=path.exists()
        checks.append({"name":name,"path":str(path),"required":required,"ok":ok,
                       "sha256":sha256(path) if ok and path.is_file() else None})
        return ok
    check("experiment freeze",root/"frozen/experiment.json")
    check("hardware audit",root/"hardware.json")
    check("model revisions",root/"frozen/model_revisions.json")
    check("budget freeze",root/"frozen/budgets.json")
    for domain in config["domains"]: check(f"manifest {domain}",root/"manifests"/f"{domain}.json")
    for model in config["models"]:
      tag=model.replace("/","__")
      check(f"support {model}",root/"support"/f"{tag}.json")
      check(f"calibration {model}",root/"calibration"/tag/"metadata.json")
      for domain in config["domains"]:
        check(f"PageDrop smoke {model} {domain}",root/"smoke"/tag/domain/"pagedrop.json")
        # Either official backend smoke is accepted; hardware selection later requires the matching one.
        v=root/"smoke"/tag/domain/"triattention_official_vllm.json"
        t=root/"smoke"/tag/domain/"triattention_official_trtllm.json"
        checks.append({"name":f"TriAttention smoke {model} {domain}","path":f"{v} OR {t}",
                       "required":True,"ok":v.exists() or t.exists(),"sha256":None})
    report={"status":"READY" if all(x["ok"] for x in checks if x["required"]) else "BLOCKED",
            "checks":checks}
    (root/"reports").mkdir(parents=True,exist_ok=True)
    (root/"reports/preflight.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True)
    a=p.parse_args(); report=audit(a.root,json.loads(a.config.read_text())); print(report["status"])
    raise SystemExit(0 if report["status"]=="READY" else 2)
if __name__=="__main__":main()
