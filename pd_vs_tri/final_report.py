from __future__ import annotations
import argparse,json
from pathlib import Path
from .progress import read_rows
from .statistics import paired

def build(root,config):
    rows=read_rows(root)
    hardware=json.loads((root/"hardware.json").read_text()) if (root/"hardware.json").exists() else {}
    budgets=json.loads((root/"frozen/budgets.json").read_text()) if (root/"frozen/budgets.json").exists() else {}
    preflight=json.loads((root/"reports/preflight.json").read_text()) if (root/"reports/preflight.json").exists() else {"status":"NOT RUN"}
    lines=["# PageDrop vs TriAttention final report","",
      "Frozen non-inferiority margin: **-3 pp**. Paired bootstrap uses 10,000 resamples and seed 0.","",
      "## Setup and provenance","",
      f"- Preflight: **{preflight.get('status','UNKNOWN')}**",
      f"- TriAttention backend: `{hardware.get('triattention_backend','UNRESOLVED')}`",
      f"- GPU: `{hardware.get('gpu_name','UNRESOLVED')}`",
      f"- FullKV-derived budget freeze: `{budgets.get('status','UNAVAILABLE')}`",
      "- PageDrop is the frozen page16, 576-to-512, recent64, prompt-protected implementation.",
      "- Scores are paired by problem; incomplete or failed items are never silently dropped.","",
      "## Cell results","",
      "| Model | Domain | Paired N | PageDrop | TriAttention | Delta | 95% CI | W/T/L | Verdict |",
      "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    complete=True; blocked=[]
    for model in config["models"]:
      for domain in config["domains"]:
        x=[r for r in rows if r.get("model")==model and r.get("domain")==domain]; s=paired(x)
        target=json.loads((root/"manifests"/f"{domain}.json").read_text())["n"]
        valid=s.get("n")==target and "bootstrap95_pp" in s; complete &= valid
        verdict="BLOCKED/INCOMPLETE" if not valid else ("NON-INFERIOR" if s["bootstrap95_pp"][0]>=-3 else "NOT ESTABLISHED")
        if not valid: blocked.append(f"{model} / {domain}: {s.get('n',0)}/{target} paired")
        lines.append(f"| {model} | {domain} | {s.get('n',0)}/{target} | {s.get('pagedrop_accuracy','—')} | {s.get('triattention_accuracy','—')} | {s.get('delta_pp','—')} | {s.get('bootstrap95_pp','—')} | {s.get('pagedrop_better',0)}/{s.get('tied',0)}/{s.get('pagedrop_worse',0)} | {verdict} |")
    failures=[r for r in rows if r.get("status") in ("error","oom")]
    elapsed=sum(float(r.get("elapsed_seconds",0)) for r in rows)
    lines += ["","## Failures and compute","",
      f"- Terminal error/OOM records: **{len(failures)}**",
      f"- Sum of per-example elapsed time: **{elapsed/3600:.2f} hours** (diagnostic; physical GPU-hours come from the job ledger).",
      f"- Blocked cells: **{len(blocked)}**"]
    lines += [f"  - {x}" for x in blocked] or ["  - None"]
    lines += ["","## Verdict","",
      ("All intended cells are complete; interpret each frozen cell verdict above." if complete else
       "INCOMPLETE/BLOCKED. No aggregate or macro non-inferiority claim is permitted."),""]
    (root/"reports").mkdir(parents=True,exist_ok=True)
    (root/"reports/pd_vs_tri_final.md").write_text("\n".join(lines)); return complete

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--config",type=Path,required=True); a=p.parse_args(); build(a.root,json.loads(a.config.read_text()))
if __name__=="__main__":main()
