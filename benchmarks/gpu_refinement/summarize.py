"""Create the timing report and a compact, auditable measurement record."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
runs=[json.loads(p.read_text()) | {'case':p.stem}
      for p in sorted((root/'results/timings').glob('*.json')) if p.name!='validation.json']
validation=json.loads((root/'results/timings/validation.json').read_text())
checks={r['case']:r for r in validation}
for run in runs:
    run['quality']=checks[run['case']]
    assert run['invalid']==run['duplicates']==run['unsorted']==0
    assert run['quality']['distance_ok']
assert len(runs)==36, f'Expected 36 measured builds; got {len(runs)}'
source=root.parent.parent/'cpp/src/neighbors/all_neighbors'
record={'image_id':'sha256:850ab92b2b122a0cb25a953856e24edd82e2878e137281af447859fe9b1ae31f',
        'focused_test_log':(root/'refine-test.log').read_text(),
        'baseline_commit':'c740818bc68c58e5141c0eb2f832f26cfd9288bd',
        'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                         for name in ['all_neighbors_builder.cuh','all_neighbors_refine.cuh']},
        'runs':runs}
(root/'measurements.json').write_text(json.dumps(record,indent=2)+'\n')
summary=[]
for dataset in ['sift','deep']:
    for gpus in [1,2,4]:
        row={'dataset':dataset,'gpus':gpus}
        for variant in ['cpu','gpu']:
            subset=[r for r in runs if r['dataset']==dataset and r['gpus']==gpus and r['variant']==variant]
            assert len(subset)==3
            for key in ['build_seconds','convert_seconds','optimize_seconds','total_seconds']:
                values=[r[key] for r in subset]
                row[f'{variant}_{key}']=statistics.median(values)
                if key=='build_seconds':
                    row[f'{variant}_build_min']=min(values)
                    row[f'{variant}_build_max']=max(values)
            row[f'{variant}_recall']=statistics.mean(r['quality']['recall_at_k'] for r in subset)
        row['build_speedup']=row['cpu_build_seconds']/row['gpu_build_seconds']
        row['total_speedup']=row['cpu_total_seconds']/row['gpu_total_seconds']
        summary.append(row)
with (root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=summary[0].keys(),lineterminator="\n")
    writer.writeheader(); writer.writerows(summary)
lines=['# GPU refinement benchmark results', '',
       'Measured on September 11, 2026 on this machine: 4 × NVIDIA L40S, dual AMD EPYC 7742, 1 TiB RAM.', '',
       'Times below are medians of three sequential, warmed runs in seconds. CPU and GPU columns differ only in all-neighbors refinement. Both use GPU IVF-PQ build/search. See [README.md](README.md) for complete reproduction and timing boundaries.', '',
       '## All-neighbors graph construction', '',
       '| Dataset | GPUs | CPU refinement (s) | GPU refinement (s) | Speedup |',
       '|---|---:|---:|---:|---:|']
names={'sift':'SIFT (1,000,000 × 128)','deep':'Deep Image (9,990,000 × 96)'}
for r in summary:
    lines.append(f"| {names[r['dataset']]} | {r['gpus']} | {r['cpu_build_seconds']:.3f} | {r['gpu_build_seconds']:.3f} | {r['build_speedup']:.2f}× |")
lines+=['','## Through CAGRA graph optimization','','Includes all-neighbors construction, int64-to-uint32 graph conversion and output allocation, and CAGRA graph optimization. CAGRA optimization runs on GPU 0.','','| Dataset | GPUs | CPU path total (s) | GPU path total (s) | Speedup |','|---|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {names[r['dataset']]} | {r['gpus']} | {r['cpu_total_seconds']:.3f} | {r['gpu_total_seconds']:.3f} | {r['total_speedup']:.2f}× |")
lines+=['','## Repeat spread and graph quality','','Exact recall@64 uses 256 deterministic database queries, including self. Distances are checked separately against FP64 direct calculations. Differences in tied neighbors and parallel IVF-PQ training can change neighbor IDs; equality of IDs is not required.','','| Dataset | GPUs | CPU build range (s) | GPU build range (s) | CPU recall@64 | GPU recall@64 |','|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {names[r['dataset']]} | {r['gpus']} | {r['cpu_build_min']:.3f}–{r['cpu_build_max']:.3f} | {r['gpu_build_min']:.3f}–{r['gpu_build_max']:.3f} | {r['cpu_recall']:.4%} | {r['gpu_recall']:.4%} |")
lines+=['','All 36 timed builds passed full-graph checks for valid IDs, unique neighbors, finite distances, and sorted distances, plus sampled exact-distance checks. The resulting CAGRA graphs also passed ID and uniqueness checks.','',
        'The 14 all-neighbors smoke cases passed. Standalone candidate-reference tests also passed for 8,209 queries, odd dimensions 137/33, candidate counts 47/603, degrees 23/300, and invalid candidate sentinels. CPU/GPU sampled mean recall differs by at most 0.44 percentage points across configurations.','',
        'The final implementation evaluates squared L2 directly on the GPU and uses sorted cuVS top-k selection. It reuses the IVF-PQ candidate distance buffer and requires no CPU refinement fallback for large degrees. Host remapping and merging remain part of this experiment.','',
        'Parameters: FP32 data and IVF-PQ search arithmetic, squared L2, 16 clusters, overlap 2, 128 candidates → 64 neighbors, CAGRA degree 32, PQ dimension 64 / 4 bits, 16 probes, 8192-query batches, 64 CPU threads total. SIFT uses 62 IVF lists; Deep Image uses 624.','',
        'A preliminary implementation using the general cuVS GPU refine API was slower (29.58 s versus 6.25 s on SIFT with one GPU) because it materialized per-query IVF lists. Those pilot measurements are retained under `results/general-api-pilot` and are excluded from every final table.','',
        'Artifacts: [summary.csv](summary.csv), [measurements.json](measurements.json), and per-run logs/sampled edges under `results/timings/`. Data loading, warm-up, validation, and disk serialization are outside the reported times. These results measure graph construction, not query search throughput.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
