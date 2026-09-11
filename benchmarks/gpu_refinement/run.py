"""Run CPU/GPU builds sequentially with a fixed 64-thread CPU budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
out = ROOT / 'results' / ('smoke' if args.smoke else 'timings')
out.mkdir(parents=True, exist_ok=True)
sift = str(ROOT / 'datasets/sift-128-euclidean/base.fbin')
deep = '/datasets/datasets/deep-image-96-inner/base.fbin'
if args.smoke:
    cases = [(v, 'sift', sift, g, 20000, c, k, m, 0)
             for v in ['cpu', 'gpu']
             for g, c, k, m in [(1,1,64,'host'), (1,1,64,'device-input'),
                                (1,1,64,'device-no-dist'), (1,4,64,'device'),
                                (2,4,64,'host'), (4,8,64,'host'),
                                (1,1,300,'host')]]
else:
    # Alternate CPU/GPU execution order on successive repeats.
    cases = [(v, name, path, g, 0, 16, 64, 'host', repeat)
             for repeat in range(args.repeats)
             for name, path in [('sift',sift),('deep',deep)]
             for g in [1,2,4]
             for v in (['cpu','gpu'] if repeat%2==0 else ['gpu','cpu'])]
for v, name, path, g, rows, clusters, k, mode, repeat in cases:
    label=f'{name}_{v}_{g}gpu_c{clusters}_k{k}_{mode}_r{repeat}'
    prefix=out / label
    if prefix.with_suffix('.json').exists():
        print('Already recorded:',label,flush=True)
        continue
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS=f'64,{64//g}', OMP_THREAD_LIMIT='128',
               OMP_MAX_ACTIVE_LEVELS='2', OMP_DYNAMIC='FALSE',
               OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES=','.join(map(str,range(g))))
    cmd=[str(ROOT/f'build/refine_{v}'),path,str(g),str(rows),str(prefix),
         str(clusters),str(k),mode]
    print(time.strftime('%H:%M:%S'),label,flush=True)
    started=time.monotonic()
    with prefix.with_suffix('.log').open('w') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print(prefix.with_suffix('.log').read_text()[-6000:],flush=True)
        raise SystemExit(f'{label}: return code {result.returncode}')
    record=json.loads(prefix.with_suffix('.json').read_text())
    record.update(variant=v,dataset=name,repeat=repeat,mode=mode,
                  process_seconds=time.monotonic()-started,omp_num_threads=env['OMP_NUM_THREADS'])
    prefix.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
