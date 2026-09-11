"""Untimed exact recall and distance checks for the sampled database queries."""
import argparse
import json
from pathlib import Path
import cupy as cp
import numpy as np

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
args=parser.parse_args()
cache={}
reports=[]
for file in sorted(args.directory.glob('*.json')):
    if file.name=='validation.json': continue
    r=json.loads(file.read_text())
    n,d,k=r['rows'],r['dim'],r['k']
    name=r.get('dataset',file.name.split('_')[0])
    dataset=(Path(r['dataset_path']) if 'dataset_path' in r else
             root/'datasets/sift-128-euclidean/base.fbin' if name=='sift' else
             Path('/datasets/datasets/deep-image-96-inner/base.fbin'))
    dtype='u1' if str(dataset).endswith('.u8bin') else '<f4'
    data=np.memmap(dataset,dtype=dtype,mode='r',offset=8,shape=(n,d))
    sample=np.fromfile(file.with_suffix('.samples'),dtype=np.dtype([
        ('row','<i8'),('ids','<i8',(k,)),('dist','<f4',(k,))]))
    key=(name,n,k)
    if key not in cache:
        queries=cp.asarray(data[sample['row']], dtype=cp.float32)
        qnorm=cp.sum(queries*queries,axis=1)[:,None]
        best_d=cp.full((len(sample),k),cp.inf,dtype=cp.float32)
        best_i=cp.full((len(sample),k),-1,dtype=cp.int64)
        for offset in range(0,n,65536):
            block=cp.asarray(data[offset:offset+65536], dtype=cp.float32)
            distances=qnorm+cp.sum(block*block,axis=1)[None,:]-2*(queries@block.T)
            # FP32 GEMM without TF32; direct distance checks below use FP64.
            take=cp.argpartition(distances,k-1,axis=1)[:,:k]
            ds=cp.take_along_axis(distances,take,axis=1)
            ids=take+offset
            both_d=cp.concatenate([best_d,ds],axis=1)
            both_i=cp.concatenate([best_i,ids],axis=1)
            take=cp.argpartition(both_d,k-1,axis=1)[:,:k]
            best_d=cp.take_along_axis(both_d,take,axis=1)
            best_i=cp.take_along_axis(both_i,take,axis=1)
        cache[key]=cp.asnumpy(best_i)
        np.savez(args.directory/f'{name}-{n}-exact.npz', rows=sample['row'], ids=cache[key])
        print('Exact reference ready:',key,flush=True)
    truth=cache[key]
    recall=np.mean([len(set(a)&set(b))/k for a,b in zip(sample['ids'],truth)])
    exact=np.sum((np.asarray(data[sample['row']],dtype=np.float64)[:,None,:]-
                  np.asarray(data[sample['ids']],dtype=np.float64))**2,axis=2)
    error=np.max(np.abs(exact-sample['dist']))
    distance_ok=(r.get('mode')=='device-no-dist' or
                 np.allclose(exact,sample['dist'],rtol=2e-4,atol=2e-4))
    report=dict(case=file.stem,recall_at_k=float(recall),max_distance_error=float(error),
                distance_ok=bool(distance_ok))
    reports.append(report)
    print(json.dumps(report),flush=True)
args.directory.joinpath('validation.json').write_text(json.dumps(reports,indent=2)+'\n')
if not all(r['distance_ok'] for r in reports): raise SystemExit('distance mismatch')
