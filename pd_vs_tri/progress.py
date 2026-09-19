from __future__ import annotations
import argparse, datetime as dt, fcntl, json, os, time
from pathlib import Path
from .config import atomic_json, canonical_hash
from .statistics import paired

def read_rows(root):
    rows=[]
    for p in sorted(root.glob("results/*/*/*/*.json")):
        try:rows.append(json.loads(p.read_text()))
        except (json.JSONDecodeError,OSError):continue
    return rows

def cell_snapshot(rows,model,domain,target):
    r=[x for x in rows if x.get("model")==model and x.get("domain")==domain]
    p=paired(r); p.update(target_n=target)
    for method in ("pagedrop16","triattention"):
        x=[z for z in r if z.get("method")==method]
        done=[z for z in x if z.get("status")=="complete"]
        p[method]={"completed":len(done),"error":sum(z.get("status")=="error" for z in x),"oom":sum(z.get("status")=="oom" for z in x),
          "mean_compression":sum(float(z.get("achieved_compression",0)) for z in done)/len(done) if done else None,
          "mean_generated_length":sum(int(z.get("generated_tokens",0)) for z in done)/len(done) if done else None,
          "median_generated_length":sorted(int(z.get("generated_tokens",0)) for z in done)[len(done)//2] if done else None}
    return p

def render(state):
    lines=["# PageDrop vs TriAttention progress","",f"Updated: `{state['timestamp']}`",f"Backend: `{state['hardware'].get('triattention_backend')}`",""]
    lines.append("| Model | Domain | Paired | PageDrop | TriAttention | Delta | 95% CI | W/T/L |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for key,c in state["cells"].items():
        model,domain=key.split("|",1); ci=c.get("bootstrap95_pp")
        lines.append(f"| {model} | {domain} | {c['n']}/{c['target_n']} | {c.get('pagedrop_accuracy','—')} | {c.get('triattention_accuracy','—')} | {c.get('delta_pp','—')} | {ci or '—'} | {c.get('pagedrop_better',0)}/{c.get('tied',0)}/{c.get('pagedrop_worse',0)} |")
    lines.extend(["",f"Physical GPU-hours: **{state['physical_gpu_hours']:.3f}**",f"Observed tokens/s: **{state.get('tokens_per_second')}**",f"ETA: **{state.get('eta')}**",""])
    return "\n".join(lines)

def update(root,config,hardware,jobs=None,force=False):
    lock_path=root/"reports/.progress.lock"; lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open("a+"); fcntl.flock(lock,fcntl.LOCK_EX)
    manifests={p.stem:json.loads(p.read_text()) for p in (root/"manifests").glob("*.json") if not p.name.endswith(".meta.json")}
    rows=read_rows(root); cells={}
    for model in config["models"]:
        for domain in config["domains"]:cells[f"{model}|{domain}"]=cell_snapshot(rows,model,domain,manifests.get(domain,{}).get("n",0))
    ledgers=[]
    for p in root.glob("runs/**/runtime_ledger.json"):
        try:ledgers.append(json.loads(p.read_text()))
        except:pass
    gpu_seconds=sum(float(x.get("active_seconds",0)) for x in ledgers)
    tokens=sum(int(r.get("generated_tokens",0)) for r in rows if r.get("status")=="complete")
    if jobs is None:
        jobs_path=root/"runs/jobs.json"
        jobs=json.loads(jobs_path.read_text()).get("jobs",[]) if jobs_path.exists() else []
    populated=[c for c in cells.values() if c.get("n",0)>=30]
    done_pairs=sum(c.get("n",0) for c in cells.values()); total_pairs=sum(c["target_n"] for c in cells.values())
    gpu_hours=gpu_seconds/3600
    eta_gpu_hours=(total_pairs-done_pairs)/(done_pairs/gpu_hours) if done_pairs and gpu_hours else None
    state={"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"code_git_sha":config.get("code_git_sha"),
      "config_hash":canonical_hash(config),"hardware":hardware,"jobs":jobs,"cells":cells,
      "physical_gpu_hours":gpu_hours,"tokens_per_second":tokens/gpu_seconds if gpu_seconds else None,
      "eta":{"rough_remaining_gpu_hours":eta_gpu_hours,"basis":"current paired-problem rate"} if eta_gpu_hours is not None else "unavailable until paired rate exists",
      "provisional_macro_delta_pp":sum(c["delta_pp"] for c in populated)/len(populated) if populated else None,
      "provisional_macro_cells":len(populated),"provisional_warning":"Incomplete-cell macro; not a final claim."}
    reports=root/"reports"; reports.mkdir(parents=True,exist_ok=True)
    atomic_json(reports/"pd_vs_tri_progress.json",state)
    tmp=reports/"pd_vs_tri_progress.md.tmp"; tmp.write_text(render(state)+"\n"); tmp.replace(reports/"pd_vs_tri_progress.md")
    with (reports/"pd_vs_tri_history.jsonl").open("a") as f:f.write(json.dumps(state,sort_keys=True)+"\n")
    compact=[]
    for key,c in cells.items():
        if c["n"]:compact.append(f"{key} | n={c['n']}/{c['target_n']} | PD={c.get('pagedrop_accuracy')} | Tri={c.get('triattention_accuracy')} | Δ={c.get('delta_pp')}pp | CI={c.get('bootstrap95_pp')} | GPUh={state['physical_gpu_hours']:.2f}")
    if compact: print("\n".join(compact),flush=True)
    fcntl.flock(lock,fcntl.LOCK_UN); lock.close(); return state

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--hardware",type=Path,required=True)
    a=p.parse_args(); update(a.root,json.loads(a.config.read_text()),json.loads(a.hardware.read_text()),force=True)
if __name__=="__main__":main()
