"""Dartmouth entry point for a fresh, hardware-consistent PageDrop pilot."""
import hashlib,json,time,sys,os
from pathlib import Path
import torch
from pagedrop.cache import PilotCache,patch_qwen,POLICIES
from kvheadroom.runtime import GPUHourBudget

ROOT=Path(os.environ.get('PD_RUN_ROOT','/outputs/pagedrop16_dartmouth_v1'))
SEED=20260915

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)

def decode(model,tok,prompts,policy,budget,forced=None):
    lengths=torch.tensor([len(p) for p in prompts],device='cuda',dtype=torch.long)
    seeds=[SEED+int(hashlib.sha256(json.dumps(p).encode()).hexdigest()[:8],16) for p in prompts]
    cache=PilotCache(model.config,lengths,policy,SEED,seeds)
    ids=torch.full((len(prompts),int(lengths.max())),tok.pad_token_id or tok.eos_token_id,device='cuda',dtype=torch.long)
    for i,p in enumerate(prompts): ids[i,:len(p)]=torch.tensor(p,device='cuda')
    positions=torch.arange(ids.shape[1],device='cuda').expand(len(prompts),-1)
    heartbeat=time.monotonic()
    def check_budget():
        nonlocal heartbeat
        budget.check()
        if time.monotonic()-heartbeat>30:
            budget.flush('generation_heartbeat'); heartbeat=time.monotonic()
    with torch.inference_mode():
        logits=model(ids,position_ids=positions,past_key_values=cache,use_cache=True).logits
        next_ids=logits[torch.arange(len(prompts),device='cuda'),lengths-1].argmax(-1)
        del logits
        if forced is not None:
            assert len({len(x) for x in forced})==1
            for step in range(len(forced[0])):
                if step%64==0: check_budget()
                inp=torch.tensor([x[step] for x in forced],device='cuda')[:,None]
                out=model(inp,position_ids=(lengths+step)[:,None],past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache.finish_step(); next_ids=out.logits[:,-1].argmax(-1)
        outputs=[[] for _ in prompts]; active=list(range(len(prompts))); step=0
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        max_steps=32 if forced is not None else 32768-int(lengths.min())
        while active and step<max_steps:
            if step%64==0: check_budget()
            values=next_ids.tolist(); survivor=[]
            for local,orig in enumerate(active):
                outputs[orig].append(values[local])
                if values[local] not in eos and (forced is not None or len(prompts[orig])+len(outputs[orig])<32768): survivor.append(local)
            if not survivor or step+1>=max_steps: break
            selection=torch.tensor(survivor,device='cuda'); next_ids=next_ids[selection]
            if len(survivor)!=len(active): cache.batch_select_indices(selection)
            active=[active[i] for i in survivor]
            position=cache.prompt_lengths[cache.rows]+cache.generated
            out=model(next_ids[:,None],position_ids=position[:,None],past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache.finish_step(); next_ids=out.logits[:,-1].argmax(-1); step+=1
    return [{'output':tok.decode(x,skip_special_tokens=True),'generated_ids':x,'generated_tokens':len(x),
             'terminated':bool(x and x[-1] in eos)} for x in outputs]


def run(commit):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from kvheadroom.traces import _prompt,_math_correct
    data_path=Path(os.environ.get('PD_MATH_DATA','/outputs/data/math/test.jsonl'))
    synth_path=Path(os.environ.get('PD_SYNTH_DATA','/outputs/kv_headroom_v1/synthetic_examples.json'))
    assert data_path.exists() and synth_path.exists(), 'Required existing artifacts missing; no replacements allowed'
    rows=[json.loads(x) for x in data_path.read_text().splitlines() if x.strip()]; assert len(rows)==500
    synth=json.loads(synth_path.read_text()); assert len(synth)>=16
    shard_id=int(os.environ.get('PD_SHARD_ID','0'))
    shards=int(os.environ.get('PD_SHARDS','1'))
    assert 0<=shard_id<shards<=16
    spec={'version':'pagedrop16-v1','model':'Qwen/Qwen3-4B','model_revision':'1cfa9a7208912126459214e8b04321603b3df60c','seed':SEED,'example_rng':'seed + sha256(prompt_ids) first32bits + layer_idx; per-example generators independent of batching/resume','policies':POLICIES,
          'generated_budget':512,'protected_recent':64,'page':16,'max_total_tokens':32768,'batch':32,
          'math_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
          'synth_sha256':hashlib.sha256(synth_path.read_bytes()).hexdigest(),'synth_ids':[x['trace_id'] for x in synth[:16]],
          'accuracy_gate':{'delta_abs_max_pp':2,'lower_ci_min_pp':-3,'random_pp_min_delta_pp':-3,
          'shared_material_drop_pp':3,'longest_quartile_drop_pp':5},'bootstrap_replicates':10000,
          'code_sha256':hashlib.sha256(b''.join(p.read_bytes() for p in sorted((Path(__file__).parent/'pagedrop').glob('*.py')))).hexdigest(),
          'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'original_cap_h100_hours':4,
          'prior_modal_h100_hours':2.39960981421889,
          'effective_l40s_gpu_hour_limit':float(os.environ.get('PD_LEDGER_HOURS','1.55')),
          'cap_override_approved':os.environ.get('PD_CAP_OVERRIDE_APPROVED')=='1',
          'backend':'flashinfer native page16 + SDPA causal prefill',
          'execution_platform':'dartmouth_l40s','fresh_all_500':True,
          'shard_id':shard_id,'shards':shards,
          'sharding':'global prompt-length batches of 32, modulo shard count; with five shards, final partial batch goes to shard 1 to balance shard 0 passcode work'}
    existing=ROOT/'request.json'
    if existing.exists():
        old=json.loads(existing.read_text()); new=json.loads(json.dumps(spec))
        if old!=new:
            previous=old.copy(); revised=new.copy(); previous.pop('code_sha256'); revised.pop('code_sha256')
            assert previous==revised and not list((ROOT/'math').glob('*/*.json')) and not list((ROOT/'sanity').glob('*/*.json')), 'Resume config mismatch after scoring'
            atomic(ROOT/('unscored_code_amendment_'+old['code_sha256'][:12]+'.json'),old)
            atomic(existing,spec); commit()
    else: atomic(existing,spec); commit()
    requested_hours=float(os.environ.get('PD_LEDGER_HOURS','1.55'))
    if requested_hours>1.55:
        assert os.environ.get('PD_CAP_OVERRIDE_APPROVED')=='1', 'Remaining original 4h cap requires explicit override'
    # A factor of one conservatively charges each physical L40S hour as one
    # H100-equivalent hour; this is a safety ledger, not a speed equivalence.
    budget=GPUHourBudget(ROOT,requested_hours,1.0,0.03)
    # Conservative per-wakeup charge covers GPU startup and up to one lost
    # 30-second heartbeat on abrupt termination; retries are disabled.
    budget.data['active_seconds']=float(budget.data['active_seconds'])+60
    budget.data['segments'].append({'label':'startup_and_crash_margin','seconds':60,'ended_unix':time.time()})
    budget.flush('precharge'); commit(); budget.check()
    with budget.active('single_worker_including_load_validation_eval_microbench'):
        tok=AutoTokenizer.from_pretrained(spec['model'],revision=spec['model_revision'])
        model=AutoModelForCausalLM.from_pretrained(spec['model'],revision=spec['model_revision'],dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
        revision=getattr(model.config,'_commit_hash',None)
        provenance={'model_revision':revision,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(), 'spec':spec}
        if (ROOT/'provenance.json').exists(): assert json.loads((ROOT/'provenance.json').read_text())['model_revision']==revision
        else: atomic(ROOT/'provenance.json',provenance)
        patch_qwen()
        # Before viewing scores, validate numerical attention equivalence and cache invariants.
        from pagedrop.validate import validate
        atomic(ROOT/'validation.json',validate(model,tok,budget)); budget.flush('validation'); commit()
        prompts=[_prompt(tok,r,'math500')[0] for r in rows]
        order=sorted(range(500),key=lambda i:(len(prompts[i]),i))
        for start in range(0,500,32):
            chunk_index=start//32
            owner=1 if shards==5 and chunk_index==15 else chunk_index%shards
            if owner!=shard_id:
                continue
            chunk=order[start:start+32]
            for policy in POLICIES:
                missing=[i for i in chunk if not (ROOT/'math'/policy/f'{i:03d}.json').exists()]
                if not missing: continue
                begun=time.monotonic()
                results=decode(model,tok,[prompts[i] for i in missing],policy,budget)
                for i,result in zip(missing,results):
                    result.update(example_id=i,policy=policy,prompt_tokens=len(prompts[i]),
                                  correct=_math_correct(result['output'],rows[i]),gold=rows[i].get('answer'),
                                  batch_seconds=time.monotonic()-begun)
                    atomic(ROOT/'math'/policy/f'{i:03d}.json',result)
                budget.flush(f'math_{start}_{policy}'); commit()
                print(json.dumps({'completed':{p:len(list((ROOT/'math'/p).glob('*.json'))) for p in POLICIES},'hours':budget.used_hours}),flush=True)
        # Existing cached fixtures: exact once-stated context, followed by original query.
        if shard_id==0:
            for policy in POLICIES:
                missing=[i for i in range(16) if not (ROOT/'sanity'/policy/f'{i:02d}.json').exists()]
                if not missing: continue
                selected=[synth[i] for i in missing]
                forced=[x['reasoning_ids'][:896]+x['query_ids'] for x in selected]
                # Query token lengths vary: group exact lengths, preserving the original fixtures.
                for length in sorted({len(x) for x in forced}):
                    which=[j for j,x in enumerate(forced) if len(x)==length]
                    results=decode(model,tok,[selected[j]['prompt_ids'] for j in which],policy,budget,[forced[j] for j in which])
                    for j,result in zip(which,results):
                        fixture=selected[j]; result.update(example_id=missing[j],policy=policy,needed_value=fixture['needed_value'],
                            correct=result['output'].lstrip().startswith(fixture['needed_value']),fixture_id=fixture['trace_id'])
                        atomic(ROOT/'sanity'/policy/f'{missing[j]:02d}.json',result)
                    budget.flush(f'sanity_{policy}'); commit()
        completed={p:len(list((ROOT/'math'/p).glob('*.json'))) for p in POLICIES}
        summary={'shard_id':shard_id,'shards':shards,'completed':completed,'sanity_done':shard_id==0}
        atomic(ROOT/'shard_complete.json',summary)
        print(json.dumps(summary),flush=True)
    commit()
    return {'run_dir':str(ROOT),'shard':json.loads((ROOT/'shard_complete.json').read_text()),'hours':budget.used_hours}


if __name__ == '__main__':
    print(json.dumps(run(lambda: None),indent=2))
