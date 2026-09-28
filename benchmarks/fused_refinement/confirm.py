import csv
import hashlib
import json
import pathlib
import subprocess
import time

root=pathlib.Path('/bench')
out=root/'runs'/'confirmation'
out.mkdir(parents=True, exist_ok=True)
meta={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [root/'bench.cu',root/'fused_refine.cuh',root/'baseline_refine.cuh',root/'bench']}}
meta['gpu_before']=subprocess.check_output(['nvidia-smi'],text=True)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
def execute(name, command):
    print('RUN',name,flush=True)
    result=subprocess.run(command,text=True,capture_output=True)
    (out/f'{name}.command.json').write_text(json.dumps(command)+'\n')
    (out/f'{name}.stdout').write_text(result.stdout)
    (out/f'{name}.stderr').write_text(result.stderr)
    if result.returncode:
        print(result.stdout,result.stderr,flush=True)
        raise SystemExit(f'{name} failed with {result.returncode}')
    return result

sanitizer='/bench/tools/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/compute-sanitizer/compute-sanitizer'
execute('memcheck',[sanitizer,'--tool','memcheck','--error-exitcode','99','/bench/bench',
    '--rows','1024','--queries','33','--dim','137','--candidates','47','--k','23','--invalid',
    '--methods','0,4,8,16,32,108,116,132','--rounds','1','--repeats','1'])
execute('racecheck',[sanitizer,'--tool','racecheck','--error-exitcode','99','/bench/bench',
    '--rows','1024','--queries','33','--dim','33','--candidates','603','--k','256','--invalid',
    '--methods','0,8,16,32,108,116,132','--rounds','1','--repeats','1'])
cases=[
 ('synthetic1m',['--queries','1048576','--methods','0,16,32,116,132']),
 ('deep1m',['--data','/datasets/deep/base.fbin','--rows','1000000','--queries','1000000','--methods','0,8,16,108,116']),
 ('deep8192',['--data','/datasets/deep/base.fbin','--rows','1000000','--queries','8192','--methods','0,8,16']),
 ('deep1m_c256_k128',['--data','/datasets/deep/base.fbin','--rows','1000000','--queries','1000000','--candidates','256','--k','128','--methods','0,8,16,108,116']),
]
rows=[]
for name,args in cases:
    result=execute(name,['/bench/bench',*args,'--rounds','11','--repeats','7'])
    for row in csv.DictReader(result.stdout.splitlines()):
        row={'case':name,**row};rows.append(row)
        print(row['method'],row['gpu_median_ms'],'ms',row['speedup'],'x',flush=True)
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta['gpu_after']=subprocess.check_output(['nvidia-smi'],text=True)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('CONFIRMATION PASSED',flush=True)
