"""Unscored numerical/structural validation, charged to the pilot cap."""
import torch
import torch.nn.functional as F
from .cache import PilotCache

def validate(model,tok,budget):
    config=model.config; device='cuda'; lengths=torch.tensor([19,31],device=device)
    h=config.num_key_value_heads; d=config.head_dim; qh=config.num_attention_heads
    checks=[]
    for policy in ('random_pp','shared_random_token','pagedrop16'):
        budget.check(); cache=PilotCache(config,lengths,policy,20260915)
        layer=cache.layers[0]
        k=torch.sin(torch.arange(2*31*h*d,device=device).reshape(2,31,h,d).float()/79).bfloat16()
        v=torch.cos(k.float()).bfloat16(); layer.initialize(k,v)
        q=torch.sin(torch.arange(2*qh*d,device=device).reshape(2,1,qh,d).float()/71).bfloat16()
        errors=[]
        for step in range(592):
            x=torch.full((2,1,h,d),float(step)/1000,device=device,dtype=torch.bfloat16)
            # Only layer 0 is needed for attention; initialize others to exercise eviction safely.
            if step==0:
                for other in cache.layers[1:]: other.initialize(k,v)
            checked_layers=(0,1) if step in (0,574,575,591) else ()
            for index,other in enumerate(cache.layers):
                if index not in checked_layers:
                    other.append(x,x); continue
                out=cache.attention(index,q,x,x)
                refs=[]
                for r,n in enumerate(lengths.tolist()):
                    ids=other.table[r,:((cache.retained+1+15)//16)]
                    gen=other.gen_k[ids].reshape(-1,h,d)[:cache.retained+1]
                    genv=other.gen_v[ids].reshape(-1,h,d)[:cache.retained+1]
                    kk=torch.cat((k[r,:n],gen),0).transpose(0,1)[None]
                    vv=torch.cat((v[r,:n],genv),0).transpose(0,1)[None]
                    refs.append(F.scaled_dot_product_attention(q[r:r+1].transpose(1,2),kk,vv,enable_gqa=True).transpose(1,2))
                ref=torch.cat(refs); err=float((out-ref).abs().max()); errors.append(err)
                assert torch.allclose(out,ref,atol=0.015,rtol=0.025), (policy,index,step,err)
            cache.finish_step()
            if step==575:
                assert cache.retained==512
                for r in range(2):
                    ids=layer.table[r,:32]; retained=layer.gen_k[ids].reshape(-1,h,d)
                    # Exact recency comparison uses the same BF16 conversion as append.
                    expected=(torch.arange(512,576,device=device).float()/1000).bfloat16()
                    assert torch.equal(retained[-64:,0,0],expected)
                    if policy=='pagedrop16':
                        assert torch.unique(ids).numel()==32
                        assert len(set(ids.tolist()) & set(layer.free[r].tolist()))==0
                        # Entire pages preserve all heads and chronological content.
                        assert torch.equal(retained[:,0],retained[:,-1])
                    elif policy=='shared_random_token': assert torch.equal(retained[:,0],retained[:,-1])
        cache.batch_select_indices(torch.tensor([1],device=device)); assert cache.rows.tolist()==[1]
        checks.append({'policy':policy,'max_attention_abs_error':max(errors),'recency_and_pool_valid':True})
    return {'passed':True,'checks':checks,'scored_generations':0}
