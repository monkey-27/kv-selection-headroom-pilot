"""Gated eviction-only CUDA-event benchmark, including selection and release.

Page release returns physical block IDs to a per-request reusable free list. The
KV pool is preallocated and remains resident, as in paged serving allocators.
Neither method includes attention or model work. Timings are per KV layer.
"""
import torch,numpy as np

def release_pages(table,keep):
    chosen=table.gather(1,keep)
    mask=torch.ones_like(table,dtype=torch.bool); mask.scatter_(1,keep,False)
    free=table[mask].view(table.shape[0],table.shape[1]-keep.shape[1])
    return chosen,free

def stable_time(fn,reset,budget):
    for _ in range(20): reset(); fn()
    torch.cuda.synchronize(); samples=[]
    for block in range(10):
        budget.check(); current=[]
        for _ in range(50):
            reset(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); current.append((start,end))
        torch.cuda.synchronize(); samples.extend(a.elapsed_time(b)*1000 for a,b in current)
        if len(samples)>=200:
            med=[np.median(samples[i:i+50]) for i in range(0,len(samples),50)]
            if max(med)/max(min(med),1e-9)<=1.1: break
    return {'median_us':float(np.median(samples)),'p25_us':float(np.percentile(samples,25)),
            'p75_us':float(np.percentile(samples,75)),'repetitions':len(samples),
            'stable_block_medians_within_10_percent':max(med)/max(min(med),1e-9)<=1.1}

def bench(config,budget):
    from kvcompress.engine.cache_utils import EvictLayer
    results=[]
    for length in (8192,16384,32768):
        for batch in (1,8,32):
            budget.check(); h=config.num_key_value_heads; d=config.head_dim
            ra=EvictLayer(0,512,64,'random_pp',0.5,False,False)
            ra.lazy_initialization(torch.empty(batch,length,h,d,device='cuda',dtype=torch.bfloat16))
            ra.k_cache.normal_(); ra.v_cache.normal_(); ra.prompt_len=0
            ra.cache_seqlens.fill_(length); ra.cumulative_length=length
            random_result=stable_time(lambda:ra._run_eviction(length),lambda:ra.cache_seqlens.fill_(length),budget)
            table=torch.arange(batch*(length//16),device='cuda').reshape(batch,-1)
            state={}
            def page_evict():
                # Same uniformly retained native old-page rule as accuracy: 28 old + 4 recent.
                old=torch.rand(batch,table.shape[1]-4,device='cuda').topk(28,-1).indices.sort(-1).values
                recent=torch.arange(table.shape[1]-4,table.shape[1],device='cuda').expand(batch,-1)
                state['table'],state['free']=release_pages(table,torch.cat((old,recent),1))
            page_result=stable_time(page_evict,lambda:None,budget)
            assert state['table'].shape==(batch,32) and state['free'].shape[1]==length//16-32
            assert set(state['table'].flatten().tolist()).isdisjoint(state['free'].flatten().tolist())
            results.append({'cache_tokens':length,'batch':batch,'random_attention':random_result,'pagedrop_native':page_result,
                            'median_speedup':random_result['median_us']/page_result['median_us']})
            del ra,table; torch.cuda.empty_cache()
    return {'scope':'per-layer eviction including random selection; excludes reset, allocation, attention, and serving throughput',
        'page_release':'native physical-page free list and retained page table; pool remains resident; no KV token gathers',
        'kv_heads':config.num_key_value_heads,'head_dim':config.head_dim,'dtype':'bfloat16','cells':results}
