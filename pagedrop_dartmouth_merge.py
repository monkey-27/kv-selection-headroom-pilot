"""Audit and combine disjoint Dartmouth PageDrop shards; apply the frozen gate."""
import argparse,hashlib,json,shutil
from pathlib import Path
from pagedrop.analyze import analyze
from pagedrop.cache import POLICIES

def main(root:Path,shards:int):
    root=root.resolve(); frozen=root/'pagedrop_frozen'/'modal_request.json'
    source=json.loads(frozen.read_text())
    code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((root/'pagedrop').glob('*.py')))).hexdigest()
    assert code==source['code_sha256'], 'policy code changed'
    for name,key in [('math500_test.jsonl','math_sha256'),('synthetic_examples.json','synth_sha256')]:
        assert hashlib.sha256((root/'pagedrop_frozen'/name).read_bytes()).hexdigest()==source[key]
    destination=root/'combined'; pending=root/'combined.pending'
    if pending.exists(): shutil.rmtree(pending)
    seen={p:set() for p in POLICIES}; provenance=[]; ledgers=[]; common_request=None
    for shard_id in range(shards):
        shard=root/'runs'/f'shard{shard_id}'
        request=json.loads((shard/'request.json').read_text())
        comparable={k:v for k,v in request.items() if k!='shard_id'}
        if common_request is None: common_request=comparable
        else: assert comparable==common_request, 'shard protocol mismatch'
        assert request['shard_id']==shard_id and request['shards']==shards
        assert request['math_sha256']==source['math_sha256'] and request['synth_sha256']==source['synth_sha256']
        assert request['code_sha256']==code and tuple(request['policies'])==POLICIES
        assert request['model_revision']==source['model_revision']
        completion=json.loads((shard/'shard_complete.json').read_text())
        assert completion['shard_id']==shard_id and completion['shards']==shards
        prov=json.loads((shard/'provenance.json').read_text()); provenance.append(prov)
        assert prov['model_revision']==source['model_revision']
        assert 'L40S' in prov['gpu'], f'unexpected GPU: {prov["gpu"]}'
        ledgers.append(json.loads((shard/'runtime_ledger.json').read_text()))
        for policy in POLICIES:
            files=sorted((shard/'math'/policy).glob('*.json'))
            assert len(files)==completion['completed'][policy]
            for file in files:
                row=json.loads(file.read_text()); i=row['example_id']
                assert isinstance(i,int) and 0<=i<500 and row['policy']==policy and i not in seen[policy]
                assert file.name==f'{i:03d}.json'
                seen[policy].add(i)
                target=pending/'math'/policy/file.name; target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target)
    for policy in POLICIES:
        assert seen[policy]==set(range(500)), f'{policy}: {len(seen[policy])}/500, missing {sorted(set(range(500))-seen[policy])[:8]}'
        for i in range(16):
            src=root/'runs'/'shard0'/'sanity'/policy/f'{i:02d}.json'
            row=json.loads(src.read_text()); assert row['example_id']==i and row['policy']==policy
            target=pending/'sanity'/policy/src.name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,target)
    report=analyze(pending)
    (pending/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (pending/'execution.json').write_text(json.dumps({
        'shards':shards,'gpu_names':[p['gpu'] for p in provenance],
        'torch_versions':[p['torch'] for p in provenance],
        'physical_l40s_gpu_hours':sum(float(x['active_seconds']) for x in ledgers)/3600,
        'prior_modal_h100_hours':2.39960981421889,
        'frozen_source_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
    },indent=2)+'\n')
    if destination.exists(): shutil.rmtree(destination)
    pending.rename(destination)
    print(json.dumps({'combined':str(destination),'completed':{p:500 for p in POLICIES},
                      'verdict':report['verdict'],'accuracy_percent':report['accuracy_percent'],
                      'benchmark_required':report['accuracy_passes']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shards',type=int,required=True)
    args=parser.parse_args(); main(args.root,args.shards)
