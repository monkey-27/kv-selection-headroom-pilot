"""Run the frozen eviction microbenchmark only after a complete accuracy GO."""
import json,os
from pathlib import Path
import numpy as np
from transformers import AutoConfig
from kvheadroom.runtime import GPUHourBudget
from pagedrop.bench import bench

root=Path(os.environ['PD_PILOT_ROOT'])/'combined'

def json_scalar(value):
    if isinstance(value,np.generic): return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')

report=json.loads((root/'report.json').read_text())
assert report['n']==500 and report['accuracy_passes'] and report['verdict']=='STRONG GO'
assert not (root/'microbenchmark.json').exists()
request=json.loads((Path(os.environ['PD_PILOT_ROOT'])/'runs'/'shard0'/'request.json').read_text())
config=AutoConfig.from_pretrained(request['model'],revision=request['model_revision'])
budget=GPUHourBudget(root,1.0,1.0,0.02)
with budget.active('gated_l40s_eviction_microbenchmark'):
    result=bench(config,budget)
target=root/'microbenchmark.json'; temp=target.with_suffix('.json.tmp')
result.update(gpu='NVIDIA L40S',trigger_verdict=report['verdict'])
temp.write_text(json.dumps(result,indent=2,default=json_scalar)+'\n'); temp.replace(target)
print(json.dumps({'benchmark':str(target),'cells':len(result['cells'])}),flush=True)
