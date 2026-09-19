"""Frozen paired analysis; refuses incomplete or mismatched cells."""
import json,itertools
from pathlib import Path
import numpy as np
from .cache import POLICIES

def analyze(root):
    root=Path(root); cells={}
    for p in POLICIES:
        rows=[json.loads((root/'math'/p/f'{i:03d}.json').read_text()) for i in range(500)]
        assert [r['example_id'] for r in rows]==list(range(500))
        cells[p]=rows
    outcomes={p:np.array([r['correct'] for r in cells[p]],dtype=int) for p in POLICIES}
    rng=np.random.default_rng(20260915); draws=rng.integers(0,500,(10000,500))
    paired={}; matrices={}
    for a,b in itertools.combinations(POLICIES,2):
        delta=outcomes[b]-outcomes[a]; ci=np.percentile(delta[draws].mean(1)*100,[2.5,97.5])
        paired[f'{b} minus {a}']={'delta_pp':round(float(delta.mean()*100),10),'bootstrap95_pp':ci.tolist()}
        matrices[f'{a} vs {b}']=[[int(((outcomes[a]==x)&(outcomes[b]==y)).sum()) for y in [0,1]] for x in [0,1]]
    patterns={''.join(map(str,bits)):int(np.logical_and.reduce([outcomes[p]==bit for p,bit in zip(POLICIES,bits)]).sum()) for bits in itertools.product([0,1],repeat=3)}
    # Shared reference length is random_pp's generated trace length. No policy-specific bins.
    length=np.array([r['generated_tokens'] for r in cells['random_pp']]); order=np.argsort(length,kind='stable')
    bins=[]
    for idx in np.array_split(order,4):
        bins.append({'n':len(idx),'reference_min':int(length[idx].min()),'reference_max':int(length[idx].max()),
            'accuracy_percent':{p:float(outcomes[p][idx].mean()*100) for p in POLICIES},
            'page_minus_shared_pp':float((outcomes['pagedrop16'][idx]-outcomes['shared_random_token'][idx]).mean()*100)})
    primary=paired['pagedrop16 minus shared_random_token']; ra=paired['pagedrop16 minus random_pp']; shared=paired['shared_random_token minus random_pp']
    shared_kill=shared['delta_pp'] < -3
    kill=shared_kill or (primary['delta_pp'] < -3 and primary['bootstrap95_pp'][1] < 0) or (ra['delta_pp'] < -3 and ra['bootstrap95_pp'][1]<0)
    passed=(not shared_kill and abs(primary['delta_pp'])<=2 and primary['bootstrap95_pp'][0]>=-3 and abs(ra['delta_pp'])<=3 and bins[-1]['page_minus_shared_pp']>=-5)
    sanity={p:[json.loads((root/'sanity'/p/f'{i:02d}.json').read_text()) for i in range(16)] for p in POLICIES}
    return {'n':500,'accuracy_percent':{p:float(outcomes[p].mean()*100) for p in POLICIES},'paired':paired,
        'disagreement_matrices':matrices,'matrix_axes':'row method A incorrect/correct; column method B incorrect/correct',
        'joint_correctness_counts':patterns,'joint_policy_order':POLICIES,'trace_length_quartiles':bins,
        'trace_lengths':{p:{'median':float(np.median([r['generated_tokens'] for r in cells[p]])),
            'max':max(r['generated_tokens'] for r in cells[p]),'unterminated':sum(not r['terminated'] for r in cells[p])} for p in POLICIES},
        'sanity':{p:{'n':16,'correct':sum(r['correct'] for r in sanity[p]),'accuracy_percent':sum(r['correct'] for r in sanity[p])*100/16} for p in POLICIES},
        'sanity_note':'existing immutable register once-stated values, cached exact pre-query context, original query, 32 greedy continuation tokens; no tuning',
        'accuracy_passes':passed,'verdict':'STRONG GO' if passed else 'KILL' if kill else 'INCONCLUSIVE',
        'shared_formulation_killed':shared_kill,'longest_trace_screen':'common random_pp longest quartile; obvious concentration defined before scores as >5 pp page minus shared degradation'}

if __name__=='__main__':
    import sys
    report=analyze(sys.argv[1]); print(json.dumps(report,indent=2))
