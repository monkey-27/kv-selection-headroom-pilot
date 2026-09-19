from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from .config import atomic_json, load, sha256

def plan(root,config_path,budget_path):
    cfg=load(config_path); budgets=json.loads(budget_path.read_text()); jobs=[]
    for model in cfg["models"]:
      for domain in cfg["domains"]:
        manifest=root/"manifests"/f"{domain}.json"; data=json.loads(manifest.read_text()); ids=[x["item_id"] for x in data["items"]]
        rng=random.Random(cfg["item_order_seed"]+int(hashlib.sha256(f"{model}|{domain}".encode()).hexdigest()[:8],16)); rng.shuffle(ids)
        key=f"{model}|{domain}"; k=budgets["cells"][key]["k"]
        for shard in range(cfg["shards_per_cell"]):
          owned=ids[shard::cfg["shards_per_cell"]]
          for method in ("pagedrop16","triattention"):
            jobs.append({"model":model,"domain":domain,"method":method,"shard":shard,"items":owned,"k":k,
              "rollout_seeds":cfg["pagedrop_seeds"] if method=="pagedrop16" else [0]})
    payload={"schema":"pd-vs-tri-plan-v1","config_sha256":sha256(config_path),"budget_sha256":sha256(budget_path),"jobs":jobs}
    atomic_json(root/"configs/execution_plan.json",payload); return payload

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--budgets",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(plan(a.root,a.config,a.budgets),indent=2))
if __name__=="__main__":main()

