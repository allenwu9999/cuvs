# BIGANN-100M GPU refinement comparison

Measured locally on September 11, 2026: 4 × NVIDIA L40S, dual AMD EPYC 7742, 1 TiB RAM. Real BIGANN prefix: 100,000,000 × 128 uint8 coordinates converted exactly to FP32. No scaling, duplication, or synthetic vectors.

One warmed run per variant and GPU count. Results are individual observations, not three-run medians like the earlier SIFT/Deep Image report. All runs execute sequentially. CPU refinement still uses GPU IVF-PQ build/search.

| GPUs | CPU refinement (s) | GPU refinement (s) | All-neighbors speedup | CPU through CAGRA (s) | GPU through CAGRA (s) | Total speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 990.092 | 325.127 | 3.05× | 1027.581 | 364.400 | 2.82× |
| 2 | 623.794 | 178.020 | 3.50× | 661.653 | 215.353 | 3.07× |
| 4 | 456.896 | 129.682 | 3.52× | 497.710 | 168.269 | 2.96× |

## Quality checks

All six graphs passed full ID-range, uniqueness, finite-distance, and distance-order checks. Final CAGRA graphs passed ID and uniqueness checks. Sampled edge distances were checked against direct FP64 calculations. Exhaustive recall@64 uses 256 database queries spread across the entire 100M rows; self is included.

| GPUs | CPU recall@64 | GPU recall@64 |
|---:|---:|---:|
| 1 | 77.8320% | 77.8381% |
| 2 | 78.5095% | 78.2959% |
| 4 | 78.4973% | 78.4851% |

## Settings and timing boundaries

- Squared L2, FP32 IVF-PQ search arithmetic, 64 clusters, overlap 2, refinement rate 2 (128 candidates → 64 neighbors), final CAGRA degree 32.
- IVF-PQ: 1,562 lists per cluster, 16 probes, PQ dimension 64, 4 bits, 10 training iterations, shape-based training fraction, 8192-query internal batch.
- Same 64-cluster partition setting across GPU counts; it differs from the 16-cluster earlier smaller-dataset experiment. These are within-dataset CPU/GPU comparisons.
- All GPUs connect through PCIe host bridges on NUMA node 0; no NVLink. CPU affinity is not explicitly bound.
- Fixed 64 OpenMP-worker budget: OMP_NUM_THREADS=64,64/gpus; dynamic teams disabled, nested parallelism enabled, passive waiting. CUDA_VISIBLE_DEVICES and RAFT ranks select exactly 1, 2, or 4 GPUs.
- Dataset loading, exact uint8-to-FP32 conversion, output allocation, and 20K-row warm-up are outside the build timer. All-neighbors timing includes clustering, IVF-PQ build/search, refinement, data transfers, and the unchanged host merge. Every timer ends with synchronization of all GPUs.
- CAGRA optimization runs on GPU 0. Its workspace uses normal device memory with the same 70% RMM builder pool. Total time adds graph conversion/allocation and CAGRA optimization to all-neighbors construction. It excludes validation and disk serialization.
- These measurements describe graph construction, not CAGRA query throughput or complete index serialization.

A preliminary CPU/1-GPU run used managed memory for pruning and was stopped after excessive paging. Its 967.45 s all-neighbors timing is excluded from the table because that run did not finish validation. The matched runs above all use device-memory pruning. The pilot log is retained in `results/bigann-100m-managed-pilot/`.

## Reproduction

Use the container setup in [README.md](README.md), then run:

```sh
python /work/benchmarks/gpu_refinement/prepare_100m.py
cmake --build /work/benchmarks/gpu_refinement/build --target refine_cpu refine_gpu -j 4
python /work/benchmarks/gpu_refinement/run_100m.py --smoke
CUDA_PATH=/opt/conda/targets/x86_64-linux CUPY_TF32=0 python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/bigann-100m-smoke
python /work/benchmarks/gpu_refinement/run_100m.py --repeats 1
CUDA_PATH=/opt/conda/targets/x86_64-linux CUPY_TF32=0 python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/bigann-100m
python /work/benchmarks/gpu_refinement/summarize_100m.py
```

The runner resumes completed cases; move the existing result directory aside to repeat from scratch. Raw logs, per-run JSON, samples, and exhaustive reference IDs are in `results/bigann-100m/`. Download provenance records checked byte ranges, per-chunk SHA-256 checksums, and the common source ETag.

Artifacts: [summary-100m.csv](summary-100m.csv), [measurements-100m.json](measurements-100m.json).
