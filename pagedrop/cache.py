"""Native page-16 cache and matched random token compaction for Qwen3.

Prompt and generated caches are separate attention partitions; merge_state combines
exact softmax states. Only post-RoPE keys are stored. PageDrop never gathers KV.
Eviction follows RA's 64-token buffered rounds, relative to generated position.
"""
import math
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer

POLICIES = ('random_pp', 'shared_random_token', 'pagedrop16')
PAGE, BUDGET, RECENT, CAPACITY = 16, 512, 64, 576


def token_keep(batch, heads, device, rng, shared=False):
    # Same independent uniform random-score topk selector as RA random_pp.
    scores = torch.stack([torch.rand(1 if shared else heads,BUDGET,device=device,generator=g) for g in rng])
    old = scores.topk(BUDGET-RECENT, dim=-1).indices.sort(-1).values
    recent = torch.arange(BUDGET, CAPACITY, device=device).view(1,1,-1).expand(batch,old.shape[1],-1)
    return torch.cat((old,recent),-1).expand(batch,heads,-1)


def page_keep(batch, device, rng):
    old = torch.stack([torch.rand(32,device=device,generator=g) for g in rng]).topk(28,-1).indices.sort(-1).values
    return torch.cat((old,torch.arange(32,36,device=device).expand(batch,-1)),1)


class PageLayer(DynamicLayer):
    def __init__(self, owner, index):
        super().__init__(); self.owner=owner; self.index=index
    def get_seq_length(self):
        return self.owner.max_prompt + self.owner.generated if self.is_initialized else 0
    def get_mask_sizes(self, cache_position):
        return self.owner.max_prompt + CAPACITY, 0
    def initialize(self,k,v):
        c=self.owner; b,_,h,d=k.shape; p=c.max_prompt_pages
        self.prompt_k=torch.zeros(b*p,PAGE,h,d,device=k.device,dtype=k.dtype)
        self.prompt_v=torch.zeros_like(self.prompt_k)
        self.gen_k=torch.zeros(b*36,PAGE,h,d,device=k.device,dtype=k.dtype)
        self.gen_v=torch.zeros_like(self.gen_k)
        for r,n in enumerate(c.prompt_lengths.tolist()):
            self.prompt_k[r*p:(r+1)*p].view(-1,h,d)[:n].copy_(k[r,:n])
            self.prompt_v[r*p:(r+1)*p].view(-1,h,d)[:n].copy_(v[r,:n])
        self.table=torch.arange(b*36,device=k.device).view(b,36)
        self.free=self.table[:,0:0]; self.rng=[torch.Generator(device=k.device).manual_seed(seed+self.index) for seed in c.example_seeds]
        self.is_initialized=True
    def append(self,k,v):
        c=self.owner; slot=c.retained; rows=c.rows
        if slot%PAGE==0 and slot>0 and c.policy=='pagedrop16' and self.free.shape[1]==4 and slot//PAGE>=32:
            # Allocate from pages released at last eviction; table only, no KV movement.
            self.table[rows,slot//PAGE]=self.free[rows,slot//PAGE-32]
        dest=self.table[rows,slot//PAGE]
        self.gen_k[dest,slot%PAGE]=k[:,0]; self.gen_v[dest,slot%PAGE]=v[:,0]
    def evict(self):
        c=self.owner; rows=c.rows; b=len(rows); h=self.gen_k.shape[2]; d=self.gen_k.shape[3]
        if c.policy=='pagedrop16':
            keep=page_keep(b,c.device,[self.rng[r] for r in rows.tolist()])
            from .bench import release_pages
            old=self.table[rows].clone(); chosen,released=release_pages(old,keep)
            self.table[rows,:32]=chosen
            if self.free.shape[1]!=4: self.free=torch.empty(c.initial_batch,4,device=c.device,dtype=torch.long)
            self.free[rows]=released
        else:
            keep=token_keep(b,h,c.device,[self.rng[r] for r in rows.tolist()],c.policy=='shared_random_token')
            # Contiguous per-head random retention is the source RA compaction operation.
            for storage in (self.gen_k,self.gen_v):
                view=storage.view(c.initial_batch,CAPACITY,h,d)
                selected=view[rows].transpose(1,2).gather(2,keep[...,None].expand(-1,-1,-1,d)).transpose(1,2)
                view[rows[:,None],torch.arange(BUDGET,device=c.device)[None,:]]=selected


class PilotCache(Cache):
    def __init__(self,config,lengths,policy,seed,example_seeds=None):
        import flashinfer
        assert policy in POLICIES
        self.policy=policy; self.seed=seed; self.prompt_lengths=lengths
        self.example_seeds=example_seeds or [seed+1009*i for i in range(len(lengths))]
        self.initial_batch=len(lengths); self.device=lengths.device
        self.rows=torch.arange(len(lengths),device=self.device)
        self.max_prompt=int(lengths.max()); self.max_prompt_pages=math.ceil(self.max_prompt/PAGE)
        self.generated=0; self.retained=0; self.config=config
        super().__init__(layers=[PageLayer(self,i) for i in range(config.num_hidden_layers)])
        self.prefix=flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(128*1024*1024,dtype=torch.uint8,device=self.device),'NHD',backend='fa2')
        self.reasoning=flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(128*1024*1024,dtype=torch.uint8,device=self.device),'NHD',backend='fa2')
        self.plan_prefix()
    def _plan(self,wrapper,table,lengths):
        counts=((lengths+PAGE-1)//PAGE).cpu()
        indptr=torch.cat((torch.zeros(1,dtype=torch.int32),counts.cumsum(0).int()))
        mask=torch.arange(table.shape[1],device=self.device)[None,:]<counts.to(self.device)[:,None]
        ids=table[mask].int()
        last=((lengths-1)%PAGE+1).int()
        wrapper.plan(indptr,ids,last,self.config.num_attention_heads,self.config.num_key_value_heads,
                     self.config.head_dim,PAGE,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,
                     pos_encoding_mode='NONE')
    def plan_prefix(self):
        table=self.rows[:,None]*self.max_prompt_pages+torch.arange(self.max_prompt_pages,device=self.device)[None,:]
        self._plan(self.prefix,table,self.prompt_lengths[self.rows])
    def plan_decode(self):
        self._plan(self.reasoning,self.layers[0].table[self.rows],torch.full((len(self.rows),),self.retained+1,device=self.device,dtype=torch.int32))
    def batch_select_indices(self,indices):
        self.rows=self.rows[indices]; self.plan_prefix()
    def finish_step(self):
        self.generated+=1; self.retained+=1
        if self.retained==CAPACITY:
            for layer in self.layers: layer.evict()
            self.retained=BUDGET
    def attention(self,index,q,k,v):
        import flashinfer
        layer=self.layers[index]
        if q.shape[1]>1 or not layer.is_initialized:
            layer.initialize(k,v)
            # Right padding cannot affect any valid query in causal prefill.
            return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                          is_causal=True,enable_gqa=True).transpose(1,2)
        layer.append(k,v)
        # Pinned FA2 planner depends on lengths, not physical IDs. Reuse its plan
        # across layers and replace only the integer page-ID buffer.
        if index==0: self.plan_decode()
        count=(self.retained+1+PAGE-1)//PAGE
        assert hasattr(self.reasoning,'_paged_kv_indices_buf')
        self.reasoning._paged_kv_indices_buf.copy_(layer.table[self.rows,:count].reshape(-1).int())
        a,s=self.prefix.run(q[:,0],(layer.prompt_k,layer.prompt_v),return_lse=True)
        b,t=self.reasoning.run(q[:,0],(layer.gen_k,layer.gen_v),return_lse=True)
        merged,_=flashinfer.merge_state(a,s,b,t)
        return merged[:,None]


def patch_qwen():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb
    def forward(self,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,**kwargs):
        shape=hidden_states.shape[:-1]; hs=(*shape,-1,self.head_dim)
        q=self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1,2)
        v=self.v_proj(hidden_states).view(hs).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        out=past_key_values.attention(self.layer_idx,q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous())
        return self.o_proj(out.reshape(*shape,-1)),None
    Qwen3Attention.forward=forward
