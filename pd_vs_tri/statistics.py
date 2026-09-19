from __future__ import annotations
import numpy as np

def aggregate_problem(rows, method):
    values=[r for r in rows if r["method"]==method and r.get("status")=="complete"]
    by={}
    for r in values:by.setdefault(r["item_id"],[]).append(float(r["correct"]))
    return {k:float(np.mean(v)) for k,v in by.items()}

def paired(rows, pd="pagedrop16", tri="triattention", reps=10000, seed=0):
    a,b=aggregate_problem(rows,pd),aggregate_problem(rows,tri); ids=sorted(set(a)&set(b))
    if not ids:return {"n":0}
    av=np.array([a[i] for i in ids]); bv=np.array([b[i] for i in ids]); d=(av-bv)*100
    out={"n":len(ids),"pagedrop_accuracy":float(av.mean()*100),"triattention_accuracy":float(bv.mean()*100),
      "delta_pp":float(d.mean()),"pagedrop_better":int((av>bv).sum()),"tied":int((av==bv).sum()),"pagedrop_worse":int((av<bv).sum())}
    if len(ids)>=30:
        rng=np.random.default_rng(seed); draws=rng.integers(0,len(ids),(reps,len(ids)))
        out["bootstrap95_pp"]=np.percentile(d[draws].mean(1),[2.5,97.5]).tolist()
    return out

