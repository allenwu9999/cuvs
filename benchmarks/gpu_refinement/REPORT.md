# GPU refinement benchmark results

Measured on September 11, 2026 on this machine: 4 × NVIDIA L40S, dual AMD EPYC 7742, 1 TiB RAM.

Times below are medians of three sequential, warmed runs in seconds. CPU and GPU columns differ only in all-neighbors refinement. Both use GPU IVF-PQ build/search. See [README.md](README.md) for complete reproduction and timing boundaries.

## All-neighbors graph construction

| Dataset | GPUs | CPU refinement (s) | GPU refinement (s) | Speedup |
|---|---:|---:|---:|---:|
| SIFT (1,000,000 × 128) | 1 | 6.555 | 3.977 | 1.65× |
| SIFT (1,000,000 × 128) | 2 | 4.542 | 2.278 | 1.99× |
| SIFT (1,000,000 × 128) | 4 | 4.596 | 1.503 | 3.06× |
| Deep Image (9,990,000 × 96) | 1 | 88.688 | 31.176 | 2.84× |
| Deep Image (9,990,000 × 96) | 2 | 63.190 | 20.519 | 3.08× |
| Deep Image (9,990,000 × 96) | 4 | 47.629 | 13.566 | 3.51× |

## Through CAGRA graph optimization

Includes all-neighbors construction, int64-to-uint32 graph conversion and output allocation, and CAGRA graph optimization. CAGRA optimization runs on GPU 0.

| Dataset | GPUs | CPU path total (s) | GPU path total (s) | Speedup |
|---|---:|---:|---:|---:|
| SIFT (1,000,000 × 128) | 1 | 6.994 | 4.327 | 1.62× |
| SIFT (1,000,000 × 128) | 2 | 4.911 | 2.655 | 1.85× |
| SIFT (1,000,000 × 128) | 4 | 4.984 | 1.852 | 2.69× |
| Deep Image (9,990,000 × 96) | 1 | 92.319 | 34.697 | 2.66× |
| Deep Image (9,990,000 × 96) | 2 | 66.997 | 24.208 | 2.77× |
| Deep Image (9,990,000 × 96) | 4 | 51.174 | 17.243 | 2.97× |

## Repeat spread and graph quality

Exact recall@64 uses 256 deterministic database queries, including self. Distances are checked separately against FP64 direct calculations. Differences in tied neighbors and parallel IVF-PQ training can change neighbor IDs; equality of IDs is not required.

| Dataset | GPUs | CPU build range (s) | GPU build range (s) | CPU recall@64 | GPU recall@64 |
|---|---:|---:|---:|---:|---:|
| SIFT (1,000,000 × 128) | 1 | 6.507–6.613 | 3.942–4.007 | 94.6309% | 94.6350% |
| SIFT (1,000,000 × 128) | 2 | 3.585–4.897 | 2.232–2.303 | 94.5251% | 94.3746% |
| SIFT (1,000,000 × 128) | 4 | 4.517–5.241 | 1.492–1.516 | 94.6086% | 94.6126% |
| Deep Image (9,990,000 × 96) | 1 | 86.889–91.531 | 29.896–31.351 | 89.3148% | 89.3962% |
| Deep Image (9,990,000 × 96) | 2 | 60.201–74.131 | 19.159–21.586 | 89.5447% | 89.1052% |
| Deep Image (9,990,000 × 96) | 4 | 45.849–59.190 | 13.332–14.827 | 89.2761% | 89.4613% |

All 36 timed builds passed full-graph checks for valid IDs, unique neighbors, finite distances, and sorted distances, plus sampled exact-distance checks. The resulting CAGRA graphs also passed ID and uniqueness checks.

The 14 all-neighbors smoke cases passed. Standalone candidate-reference tests also passed for 8,209 queries, odd dimensions 137/33, candidate counts 47/603, degrees 23/300, and invalid candidate sentinels. CPU/GPU sampled mean recall differs by at most 0.44 percentage points across configurations.

The final implementation evaluates squared L2 directly on the GPU and uses sorted cuVS top-k selection. It reuses the IVF-PQ candidate distance buffer and requires no CPU refinement fallback for large degrees. Host remapping and merging remain part of this experiment.

Parameters: FP32 data and IVF-PQ search arithmetic, squared L2, 16 clusters, overlap 2, 128 candidates → 64 neighbors, CAGRA degree 32, PQ dimension 64 / 4 bits, 16 probes, 8192-query batches, 64 CPU threads total. SIFT uses 62 IVF lists; Deep Image uses 624.

A preliminary implementation using the general cuVS GPU refine API was slower (29.58 s versus 6.25 s on SIFT with one GPU) because it materialized per-query IVF lists. Those pilot measurements are retained under `results/general-api-pilot` and are excluded from every final table.

Artifacts: [summary.csv](summary.csv), [measurements.json](measurements.json), and per-run logs/sampled edges under `results/timings/`. Data loading, warm-up, validation, and disk serialization are outside the reported times. These results measure graph construction, not query search throughput.
