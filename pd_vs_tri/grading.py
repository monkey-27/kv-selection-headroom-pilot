from __future__ import annotations
import json, os, re, subprocess, tempfile

def _last_boxed(text):
    hits=re.findall(r"\\boxed\{([^{}]+)\}",text)
    return hits[-1].strip() if hits else text.strip().splitlines()[-1].strip()

def grade(completion,item):
    spec=item.get("grading") or {}; kind=spec.get("kind")
    if kind=="exact": return _last_boxed(completion)==str(spec["answer"]).strip()
    if kind=="numeric":
        try:return abs(float(_last_boxed(completion))-float(spec["answer"]))<=float(spec.get("tolerance",0))
        except:return False
    if kind=="multiple_choice":
        hit=re.findall(r"\b([A-Z])\b",_last_boxed(completion).upper()); return bool(hit) and hit[-1]==str(spec["answer"]).upper()
    if kind=="code":
        command=os.environ.get("PDVT_OFFICIAL_CODE_GRADER")
        if not command:raise RuntimeError("official code grader command is required")
        payload={"completion":completion,"item":item}
        p=subprocess.run([command],input=json.dumps(payload),text=True,capture_output=True,check=True)
        result=json.loads(p.stdout); assert result.get("official_grader") is True
        return bool(result["correct"])
    raise RuntimeError(f"unsupported or missing objective grader: {kind!r}")
