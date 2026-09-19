from __future__ import annotations
import json, platform, subprocess, sys, time

def audit() -> dict:
    out={"timestamp":time.time(),"hostname":platform.node(),"python":sys.version,"cuda_available":False}
    try:
        import torch
        out.update(torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p=torch.cuda.get_device_properties(0)
            out.update(gpu=p.name,total_memory=int(p.total_memory),compute_capability=[p.major,p.minor])
            out["triattention_backend"]="tensorrt_llm" if (p.major==10 and p.minor in (0,3)) else "triattention_official_vllm"
        else: out["triattention_backend"]="unavailable"
    except Exception as e: out["torch_error"]=repr(e); out["triattention_backend"]="unavailable"
    for name,cmd in {"nvidia_smi":["nvidia-smi","--query-gpu=name,compute_cap","--format=csv,noheader"],"git":["git","--version"]}.items():
        try: out[name]=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=15).stdout.strip()
        except Exception as e: out[name+"_error"]=repr(e)
    return out

if __name__=="__main__": print(json.dumps(audit(),indent=2))

