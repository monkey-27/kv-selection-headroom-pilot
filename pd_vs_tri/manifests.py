from __future__ import annotations
import argparse, datetime as dt, hashlib, json, random
from pathlib import Path
from .config import atomic_json, sha256

CUTOFF=dt.date(2025,4,30)
EXPECTED={"math":113,"constraint":200,"science":100}

def _date(value):
    if not value: raise ValueError("missing original publication date")
    return dt.date.fromisoformat(str(value)[:10])

def _read(path):
    if path.suffix==".jsonl": return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    data=json.loads(path.read_text()); return data if isinstance(data,list) else data["items"]

def _difficulty(row):
    value=row.get("difficulty",row.get("rating",0))
    try:return float(value)
    except:return 0.0

def stratified_lcb(rows,n=200,seed=0):
    eligible=[r for r in rows if _date(r.get("original_publication_date"))>CUTOFF]
    eligible.sort(key=lambda r:(_date(r["original_publication_date"]),_difficulty(r),str(r.get("id"))))
    if len(eligible)<=n:return eligible
    # Deterministic round-robin over date/difficulty bins; never inspect outcomes.
    dates=sorted({_date(r["original_publication_date"]) for r in eligible})
    def dbin(r): return min(3,dates.index(_date(r["original_publication_date"]))*4//len(dates))
    vals=sorted(_difficulty(r) for r in eligible)
    def qbin(r): return min(3,sum(v<=_difficulty(r) for v in vals)*4//(len(vals)+1))
    buckets={}
    for r in eligible:buckets.setdefault((dbin(r),qbin(r)),[]).append(r)
    rng=random.Random(seed)
    for b in buckets.values():rng.shuffle(b)
    picked=[]
    while len(picked)<n:
        moved=False
        for key in sorted(buckets):
            if buckets[key] and len(picked)<n:picked.append(buckets[key].pop()); moved=True
        if not moved:break
    return sorted(picked,key=lambda r:str(r.get("id")))

def build(domain,inputs,out,source_meta):
    raw=[]
    for path in inputs:
        for r in _read(path):
            x=dict(r); x["source_file"]=path.name; raw.append(x)
    if domain=="code": raw=stratified_lcb(raw)
    if domain in EXPECTED and len(raw)!=EXPECTED[domain]: raise ValueError(f"{domain}: expected {EXPECTED[domain]}, got {len(raw)}")
    ids=[]; items=[]
    for i,r in enumerate(raw):
        item_id=str(r.get("item_id",r.get("id",f"{domain}-{i:04d}")))
        if item_id in ids:raise ValueError(f"duplicate item_id {item_id}")
        if domain=="code" and _date(r.get("original_publication_date"))<=CUTOFF:raise ValueError(f"pre-cutoff code item {item_id}")
        ids.append(item_id); items.append({"item_id":item_id,"domain":domain,"source":r.get("source",source_meta["source"]),
          "source_revision":source_meta["revision"],"original_publication_date":r.get("original_publication_date"),
          "question":r.get("question",r.get("problem",r.get("prompt"))),"grading":r.get("grading",{}),"raw":r})
    payload={"schema":"pd-vs-tri-manifest-v1","domain":domain,"n":len(items),"source":source_meta,"items":items}
    atomic_json(out,payload); atomic_json(out.with_suffix(".meta.json"),{"manifest_sha256":sha256(out),"inputs":{str(p):sha256(p) for p in inputs}})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domain",required=True); p.add_argument("--input",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--source",required=True); p.add_argument("--revision",required=True)
    a=p.parse_args(); build(a.domain,a.input,a.output,{"source":a.source,"revision":a.revision})
if __name__=="__main__":main()

