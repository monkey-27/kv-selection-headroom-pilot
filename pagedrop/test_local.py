"""Unscored CPU checks for resume stability, physical page ownership, paired gates."""
import json,tempfile,unittest
from pathlib import Path
import torch
from .cache import token_keep,page_keep,POLICIES
from .bench import release_pages
from .analyze import analyze

class LocalChecks(unittest.TestCase):
    def test_token_rng_and_head_sharing(self):
        for shared in (False,True):
            rng=[torch.Generator().manual_seed(20260915+i) for i in range(3)]
            keep=token_keep(3,8,'cpu',rng,shared)
            self.assertEqual(tuple(keep.shape),(3,8,512))
            self.assertTrue(torch.equal(keep[:,:,-64:],torch.arange(512,576).expand(3,8,-1)))
            self.assertTrue(all(len(set(x.tolist()))==512 for x in keep.reshape(-1,512)))
            self.assertEqual(torch.equal(keep[:,0],keep[:,1]),shared)
            resumed=token_keep(1,8,'cpu',[torch.Generator().manual_seed(20260915)],shared)
            self.assertTrue(torch.equal(resumed[0],keep[0]))
    def test_page_release_partition(self):
        for pages in (36,512,1024,2048):
            table=torch.arange(3*pages).reshape(3,pages)
            old=torch.arange(28).expand(3,-1)
            keep=torch.cat((old,torch.arange(pages-4,pages).expand(3,-1)),1)
            retained,free=release_pages(table,keep)
            self.assertEqual(tuple(retained.shape),(3,32))
            self.assertEqual(tuple(free.shape),(3,pages-32))
            for r in range(3):
                self.assertFalse(set(retained[r].tolist()) & set(free[r].tolist()))
                self.assertEqual(set(retained[r].tolist())|set(free[r].tolist()),set(table[r].tolist()))
    def test_paired_gate_and_incomplete_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def write(drop=0,shared_drop=0):
                for p in POLICIES:
                    (root/'math'/p).mkdir(parents=True,exist_ok=True); (root/'sanity'/p).mkdir(parents=True,exist_ok=True)
                    for i in range(500):
                        loss=drop if p=='pagedrop16' else shared_drop if p=='shared_random_token' else 0
                        row={'example_id':i,'correct':i<400-loss,'generated_tokens':512+i,'terminated':True}
                        (root/'math'/p/f'{i:03d}.json').write_text(json.dumps(row))
                    for i in range(16): (root/'sanity'/p/f'{i:02d}.json').write_text(json.dumps({'correct':False}))
            write(); report=analyze(root)
            self.assertEqual(report['verdict'],'STRONG GO')
            self.assertEqual(report['paired']['pagedrop16 minus shared_random_token']['bootstrap95_pp'],[0,0])
            self.assertEqual(report['joint_correctness_counts']['111'],400)
            self.assertEqual(report['disagreement_matrices']['random_pp vs pagedrop16'],[[100,0],[0,400]])
            write(drop=40); self.assertEqual(analyze(root)['verdict'],'KILL')
            write(drop=40,shared_drop=40); self.assertTrue(analyze(root)['shared_formulation_killed'])
            (root/'math'/'pagedrop16'/'499.json').unlink()
            with self.assertRaises(FileNotFoundError): analyze(root)

if __name__=='__main__': unittest.main()
