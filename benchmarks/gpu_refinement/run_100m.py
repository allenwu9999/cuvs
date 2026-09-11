"""Matched real BIGANN-100M runs; keep earlier SIFT/Deep results untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--repeats',type=int,default=1)
p.add_argument('--smoke',action='store_true')
a=p.parse_args()
output=root/'results'/('bigann-100m-smoke' if a.smoke else 'bigann-100m')
output.mkdir(parents=True,exist_ok=True)
data=root/'datasets/bigann-100m/base.100M.u8bin'
for repeat in range(a.repeats):
    for gpus in [1,2,4]:
        for variant in (['gpu','cpu'] if repeat%2==0 else ['cpu','gpu']):
            label=f'bigann100m_{variant}_{gpus}gpu_r{repeat}'
            prefix=output/label
            if prefix.with_suffix('.json').exists():
                r=json.loads(prefix.with_suffix('.json').read_text())
                assert r.get('variant')==variant and r['invalid']==r['duplicates']==r['unsorted']==0
                print('Already recorded',label,flush=True)
                continue
            env=os.environ.copy()
            env.update(OMP_NUM_THREADS=f'64,{64//gpus}', OMP_THREAD_LIMIT='128',
                       OMP_MAX_ACTIVE_LEVELS='2', OMP_DYNAMIC='FALSE',
                       OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES=','.join(map(str,range(gpus))))
            rows=100000 if a.smoke else 100000000
            cmd=[str(root/f'build/refine_{variant}'),str(data),str(gpus),str(rows),
                 str(prefix),'64','64','host','device']
            print(time.strftime('%Y-%m-%d %H:%M:%S'),label,flush=True)
            start=time.monotonic()
            with prefix.with_suffix('.log').open('w') as log:
                proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                print(prefix.with_suffix('.log').read_text()[-6000:],flush=True)
                raise SystemExit(f'{label} failed: {proc.returncode}')
            r=json.loads(prefix.with_suffix('.json').read_text())
            r.update(variant=variant,dataset='bigann100m',repeat=repeat,mode='host',
                     dataset_path=str(data),input_dtype='uint8 converted exactly to FP32',
                     omp_num_threads=env['OMP_NUM_THREADS'],process_seconds=time.monotonic()-start)
            prefix.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
            print(json.dumps(r),flush=True)
