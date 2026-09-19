from __future__ import annotations
import argparse, json, os, subprocess, time, traceback
from pathlib import Path
from .config import atomic_json, sha256
from .official_triattention import fail_closed_support
from .grading import grade

def verified(path,item,job):
    if not path.exists():return False
    try:r=json.loads(path.read_text())
    except:return False
    return r.get("item_id")==item["item_id"] and r.get("method")==job["method"] and r.get("status") in ("complete","error","oom")

def external_generate(job,item,seed,root,max_attempts=3):
    # Backends write a single canonical JSON result. This wrapper intentionally
    # contains no substitute attention implementation.
    command=os.environ.get("PDVT_GENERATE_COMMAND")
    if not command:raise RuntimeError("PDVT_GENERATE_COMMAND is required on Dartmouth")
    request=root/"tmp"/f"{item['item_id']}.{job['method']}.{seed}.request.json"
    response=request.with_suffix(".response.json")
    atomic_json(request,{"job":job,"item":item,"seed":seed,"response_path":str(response)})
    last=None
    for attempt in range(1,max_attempts+1):
      try:
        subprocess.run([command,str(request)],check=True)
        result=json.loads(response.read_text())
        response.unlink(missing_ok=True); request.unlink(missing_ok=True)
        return result
      except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        last=exc
        if attempt < max_attempts:
          time.sleep(2**(attempt-1))
    raise RuntimeError(f"generator failed after {max_attempts} attempts: {last}")

def persist_raw(root,job,item_id,seed,result):
    rawdir=root/"raw_completions"/job["model"].replace("/","__")/job["domain"]/job["method"]
    raw=rawdir/f"{item_id}.seed{seed}.json"
    payload={"item_id":item_id,"model":job["model"],"domain":job["domain"],
      "method":job["method"],"seed":seed,"raw_completion":result["raw_completion"],
      "backend_metadata":result.get("backend_metadata",{})}
    atomic_json(raw,payload)
    return raw

def run(root,job_path,manifest_path,hardware_path):
    job=json.loads(job_path.read_text()); manifest=json.loads(manifest_path.read_text()); hardware=json.loads(hardware_path.read_text())
    items={x["item_id"]:x for x in manifest["items"]}
    if job["method"]=="triattention":
        backend=hardware["triattention_backend"]
        fail_closed_support(job["model"],backend,root/"smoke"/job["model"].replace("/","__")/job["domain"]/f"{backend}.json")
    outdir=root/"results"/job["model"].replace("/","__")/job["domain"]/job["method"]
    newly=0; last_progress=time.monotonic()
    for item_id in job["items"]:
      item=items[item_id]
      for seed in job["rollout_seeds"]:
        out=outdir/f"{item_id}.seed{seed}.json"
        if verified(out,item,job):continue
        begun=time.time()
        try:
          result=external_generate(job,item,seed,root)
          if "raw_completion" not in result:raise RuntimeError("backend omitted raw_completion")
          raw_path=persist_raw(root,job,item_id,seed,result)
          result["correct"]=bool(grade(result["raw_completion"],item)); result.update(status="complete")
          result["raw_artifact"]=str(raw_path.relative_to(root)); result["raw_sha256"]=sha256(raw_path)
        except RuntimeError as e:
          if "out of memory" in str(e).lower():result={"status":"oom","error":str(e)}
          else:result={"status":"error","error":str(e),"traceback":traceback.format_exc()}
        result.update(item_id=item_id,model=job["model"],domain=job["domain"],method=job["method"],seed=seed,
          budget_k=job["k"],elapsed_seconds=time.time()-begun,manifest_sha256=sha256(manifest_path))
        atomic_json(out,result)
        newly+=1
        if newly>=25 or time.monotonic()-last_progress>=600:
          resolved=root/"configs/pd_vs_tri_headline.resolved.json"
          if resolved.exists():
            from .progress import update
            update(root,json.loads(resolved.read_text()),hardware)
          newly=0; last_progress=time.monotonic()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--job",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--hardware",type=Path,required=True); a=p.parse_args(); run(a.root,a.job,a.manifest,a.hardware)
if __name__=="__main__":main()
