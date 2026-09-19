from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .config import atomic_json, sha256

def choose_k(lengths):
    x=np.asarray(lengths,dtype=np.int64); assert x.size and np.all(x>0)
    candidates=np.arange(16,int(x.max())+16,16)
    ratios=np.minimum(x[:,None],candidates[None,:]).mean(0)/x.mean()
    idx=int(np.argmin(np.abs(ratios-.25)))
    return int(candidates[idx]),float(ratios[idx])

def freeze(trace_files,out):
    if out.exists(): raise FileExistsError(f"budget freeze already exists: {out}")
    cells={}
    for path in trace_files:
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        keys={(r["model"],r["domain"]) for r in rows}; assert len(keys)==1
        model,domain=keys.pop(); lengths=[int(r["generated_tokens"]) for r in rows]
        k,ratio=choose_k(lengths); cells[f"{model}|{domain}"]={"k":k,"achieved_ratio_on_fullkv":ratio,"n":len(lengths),"trace_sha256":sha256(path)}
    atomic_json(out,{"schema":"headline-budget-freeze-v1","selection_uses":"FullKV generated lengths only","target_ratio":0.25,"cells":cells})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--trace",type=Path,action="append",required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); freeze(a.trace,a.output)
if __name__=="__main__":main()
