"""Summarize the separate 100M-vector experiment without changing earlier reports."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
folder=root/'results/bigann-100m'
runs=[json.loads(p.read_text()) | {'case':p.stem} for p in sorted(folder.glob('*.json'))
      if p.name!='validation.json']
quality={r['case']:r for r in json.loads((folder/'validation.json').read_text())}
assert len(runs)>=6
for r in runs:
    assert r['rows']==100_000_000 and r['clusters']==64 and r['k']==64
    assert r['optimizer_memory']=='device'
    assert r['invalid']==r['duplicates']==r['unsorted']==0
    r['quality']=quality[r['case']]
    assert r['quality']['distance_ok']
source=root.parent.parent/'cpp/src/neighbors/all_neighbors'
record=dict(baseline_commit='c740818bc68c58e5141c0eb2f832f26cfd9288bd',
            image_id='sha256:850ab92b2b122a0cb25a953856e24edd82e2878e137281af447859fe9b1ae31f',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [source/'all_neighbors_builder.cuh',source/'all_neighbors_refine.cuh']},
            harness_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [root/'main.cu',root/'CMakeLists.txt',root/'run_100m.py',root/'validate.py']},
            hardware=dict(gpus='4 x NVIDIA L40S',cpu='2 x AMD EPYC 7742',ram='1 TiB',
                          gpu_topology='All four GPUs on NUMA node 0, PCIe NODE links; no NVLink',
                          cpu_affinity='No explicit binding; fixed OpenMP thread budget'),
            dataset=json.loads((root/'datasets/bigann-100m/download-100000000.json').read_text()),
            runs=runs)
(root/'measurements-100m.json').write_text(json.dumps(record,indent=2)+'\n')
rows=[]
for gpus in [1,2,4]:
    row={'gpus':gpus}
    for variant in ['cpu','gpu']:
        subset=[r for r in runs if r['gpus']==gpus and r['variant']==variant]
        assert subset
        row[f'{variant}_runs']=len(subset)
        for key in ['build_seconds','convert_seconds','optimize_seconds','total_seconds']:
            row[f'{variant}_{key}']=statistics.median(r[key] for r in subset)
        row[f'{variant}_recall']=statistics.mean(r['quality']['recall_at_k'] for r in subset)
    row['build_speedup']=row['cpu_build_seconds']/row['gpu_build_seconds']
    row['total_speedup']=row['cpu_total_seconds']/row['gpu_total_seconds']
    rows.append(row)
with (root/'summary-100m.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(rows)
lines=['# BIGANN-100M GPU refinement comparison','',
       'Measured locally on September 11, 2026: 4 × NVIDIA L40S, dual AMD EPYC 7742, 1 TiB RAM. Real BIGANN prefix: 100,000,000 × 128 uint8 coordinates converted exactly to FP32. No scaling, duplication, or synthetic vectors.','',
       'One warmed run per variant and GPU count. Results are individual observations, not three-run medians like the earlier SIFT/Deep Image report. All runs execute sequentially. CPU refinement still uses GPU IVF-PQ build/search.','',
       '| GPUs | CPU refinement (s) | GPU refinement (s) | All-neighbors speedup | CPU through CAGRA (s) | GPU through CAGRA (s) | Total speedup |',
       '|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['gpus']} | {r['cpu_build_seconds']:.3f} | {r['gpu_build_seconds']:.3f} | {r['build_speedup']:.2f}× | {r['cpu_total_seconds']:.3f} | {r['gpu_total_seconds']:.3f} | {r['total_speedup']:.2f}× |")
lines+=['','## Quality checks','','All six graphs passed full ID-range, uniqueness, finite-distance, and distance-order checks. Final CAGRA graphs passed ID and uniqueness checks. Sampled edge distances were checked against direct FP64 calculations. Exhaustive recall@64 uses 256 database queries spread across the entire 100M rows; self is included.','','| GPUs | CPU recall@64 | GPU recall@64 |','|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['gpus']} | {r['cpu_recall']:.4%} | {r['gpu_recall']:.4%} |")
lines+=['','## Settings and timing boundaries','',
        '- Squared L2, FP32 IVF-PQ search arithmetic, 64 clusters, overlap 2, refinement rate 2 (128 candidates → 64 neighbors), final CAGRA degree 32.',
        '- IVF-PQ: 1,562 lists per cluster, 16 probes, PQ dimension 64, 4 bits, 10 training iterations, shape-based training fraction, 8192-query internal batch.',
        '- Same 64-cluster partition setting across GPU counts; it differs from the 16-cluster earlier smaller-dataset experiment. These are within-dataset CPU/GPU comparisons.',
        '- All GPUs connect through PCIe host bridges on NUMA node 0; no NVLink. CPU affinity is not explicitly bound.',
        '- Fixed 64 OpenMP-worker budget: OMP_NUM_THREADS=64,64/gpus; dynamic teams disabled, nested parallelism enabled, passive waiting. CUDA_VISIBLE_DEVICES and RAFT ranks select exactly 1, 2, or 4 GPUs.',
        '- Dataset loading, exact uint8-to-FP32 conversion, output allocation, and 20K-row warm-up are outside the build timer. All-neighbors timing includes clustering, IVF-PQ build/search, refinement, data transfers, and the unchanged host merge. Every timer ends with synchronization of all GPUs.',
        '- CAGRA optimization runs on GPU 0. Its workspace uses normal device memory with the same 70% RMM builder pool. Total time adds graph conversion/allocation and CAGRA optimization to all-neighbors construction. It excludes validation and disk serialization.',
        '- These measurements describe graph construction, not CAGRA query throughput or complete index serialization.','',
        'A preliminary CPU/1-GPU run used managed memory for pruning and was stopped after excessive paging. Its 967.45 s all-neighbors timing is excluded from the table because that run did not finish validation. The matched runs above all use device-memory pruning. The pilot log is retained in `results/bigann-100m-managed-pilot/`.','',
        '## Reproduction','',
        'Use the container setup in [README.md](README.md), then run:','',
        '```sh',
        'python /work/benchmarks/gpu_refinement/prepare_100m.py',
        'cmake --build /work/benchmarks/gpu_refinement/build --target refine_cpu refine_gpu -j 4',
        'python /work/benchmarks/gpu_refinement/run_100m.py --smoke',
        'CUDA_PATH=/opt/conda/targets/x86_64-linux CUPY_TF32=0 python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/bigann-100m-smoke',
        'python /work/benchmarks/gpu_refinement/run_100m.py --repeats 1',
        'CUDA_PATH=/opt/conda/targets/x86_64-linux CUPY_TF32=0 python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/bigann-100m',
        'python /work/benchmarks/gpu_refinement/summarize_100m.py',
        '```','',
        'The runner resumes completed cases; move the existing result directory aside to repeat from scratch. Raw logs, per-run JSON, samples, and exhaustive reference IDs are in `results/bigann-100m/`. Download provenance records checked byte ranges, per-chunk SHA-256 checksums, and the common source ETag.','',
        'Artifacts: [summary-100m.csv](summary-100m.csv), [measurements-100m.json](measurements-100m.json).']
(root/'REPORT-100M.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
