import json
from pathlib import Path
import numpy as np
import pytest
from pd_vs_tri.budgets import choose_k
from pd_vs_tri.manifests import stratified_lcb, build
from pd_vs_tri.statistics import aggregate_problem, paired
from pd_vs_tri.grading import grade
from pd_vs_tri.preflight import audit
from pd_vs_tri.runner import persist_raw

def test_budget_is_page_aligned_and_uses_only_lengths():
    k,ratio=choose_k([64,128,256,512])
    assert k%16==0
    candidates=range(16,513,16)
    best=min(candidates,key=lambda x:abs(np.mean(np.minimum([64,128,256,512],x))/240-.25))
    assert k==best and 0<ratio<1

def test_pagedrop_seeds_are_aggregated_before_pairing():
    rows=[
      {"item_id":"a","method":"pagedrop16","status":"complete","correct":1},
      {"item_id":"a","method":"pagedrop16","status":"complete","correct":0},
      {"item_id":"a","method":"triattention","status":"complete","correct":0},
      {"item_id":"b","method":"pagedrop16","status":"complete","correct":1},
      {"item_id":"b","method":"pagedrop16","status":"complete","correct":1},
      {"item_id":"b","method":"triattention","status":"complete","correct":1}]
    assert aggregate_problem(rows,"pagedrop16")=={"a":.5,"b":1.0}
    result=paired(rows); assert result["n"]==2 and result["delta_pp"]==25

def test_lcb_strict_cutoff_and_determinism():
    rows=[{"id":str(i),"original_publication_date":"2025-05-01","difficulty":i%5} for i in range(250)]
    rows += [{"id":"old","original_publication_date":"2025-04-30","difficulty":5}]
    a=stratified_lcb(rows); b=stratified_lcb(rows)
    assert len(a)==200 and [x["id"] for x in a]==[x["id"] for x in b]
    assert all(x["id"]!="old" for x in a)

def test_manifest_fails_closed_on_wrong_expected_size(tmp_path):
    src=tmp_path/"x.jsonl"; src.write_text(json.dumps({"id":"x","question":"q"})+"\n")
    with pytest.raises(ValueError):build("constraint",[src],tmp_path/"out.json",{"source":"official","revision":"sha"})

def test_objective_graders():
    assert grade("work\n\\boxed{42}",{"grading":{"kind":"exact","answer":"42"}})
    assert grade("B",{"grading":{"kind":"multiple_choice","answer":"B"}})

def test_preflight_fails_closed(tmp_path):
    report=audit(tmp_path,{"models":["org/model"],"domains":["math"]})
    assert report["status"]=="BLOCKED"
    assert (tmp_path/"reports/preflight.json").exists()

def test_raw_completion_is_durable(tmp_path):
    job={"model":"org/model","domain":"math","method":"pagedrop"}
    path=persist_raw(tmp_path,job,"problem-1",20260915,
      {"raw_completion":"answer","backend_metadata":{"revision":"abc"}})
    row=json.loads(path.read_text())
    assert row["raw_completion"]=="answer"
    assert row["backend_metadata"]["revision"]=="abc"
