# Fused refinement: measured improvement on this machine

Measured September 28, 2026, on one **NVIDIA RTX PRO 6000 Blackwell Server Edition
(96 GB)**, CUDA 12.9, driver 580.173.02.

**The fused prototype is 1.42x faster on one million Deep Image queries with
128 IVF-PQ candidates refined to 64 neighbors, even when both implementations
use the same 8192-query batching. Launching the fused kernel over all queries
raises the speedup to 1.60x.**

These are refinement-only measurements against the existing custom two-stage
GPU path, not against CPU refinement or built-in IVF-Flat-based GPU refinement.
They do not measure complete graph construction.

## Confirmed results

Times are milliseconds per complete refinement call, medians of 11 rounds with
7 calls per round. All input data and candidate IDs are identical across methods
within each case, with allocations and IVF-PQ build/search outside timing.

### Same 8192-query batching on both paths

| Dataset / query count | Dimensions | Candidates -> neighbors | Existing two-stage GPU | Fused GPU | Speedup |
|---|---:|---:|---:|---:|---:|
| Deep Image / 1,000,000 | 96 | 128 -> 64 | 30.605 ms | 21.541 ms | **1.42x** |
| Deep Image / 1,000,000 | 96 | 256 -> 128 | 61.348 ms | 46.615 ms | **1.32x** |
| Deep Image / 8,192 | 96 | 128 -> 64 | 0.250 ms | 0.180 ms | **1.39x** |
| Synthetic / 1,048,576 | 128 | 128 -> 64 | 41.890 ms | 38.762 ms | **1.08x** |

Deep Image uses 8 threads per candidate. Synthetic uses 16 threads per candidate
in this equal-batching table. The initial tuning sweep selected these configurations;
all measured alternatives are retained in the CSV rather than reporting only winners.

### Allow one fused launch for all queries

The fused implementation needs no global candidate-distance buffer or separate
selection workspace. It therefore does not require the original 8192-query
selection batches; one CTA per query can be launched across the full workload.
The baseline remains unchanged.

| Dataset / query count | Candidates -> neighbors | Existing two-stage GPU | Fused, one launch | Speedup |
|---|---:|---:|---:|---:|
| Deep Image / 1,000,000 | 128 -> 64 | 30.605 ms | 19.079 ms | **1.60x** |
| Deep Image / 1,000,000 | 256 -> 128 | 61.348 ms | 41.240 ms | **1.49x** |
| Synthetic / 1,048,576 | 128 -> 64 | 41.890 ms | 36.705 ms | **1.14x** |

The last row uses 32 threads per candidate; 16 threads performs almost identically
(36.746 ms). These results demonstrate that the gain depends on data and scheduling,
and is not a universal multiplier. All principal cases use refinement ratio 2.

For the primary Deep Image case, observed per-round ranges were 30.594–30.615 ms
for the baseline, 21.539–21.545 ms for batched fused, and 19.077–19.084 ms for
single-launch fused. These are observed ranges, not statistical confidence intervals.

## What changed

The original kernel assigns a warp to each query/candidate pair, materializes the
candidate distances in global memory, then calls `cuvs::selection::select_k`.
The prototype instead:

1. Assigns a 128-thread block to each query and shares its coordinates in shared memory.
2. Reads candidate vectors directly, using configurable thread teams along dimensions.
3. Keeps a 128-distance tile and warp top-k queues on chip.
4. Merges the queues with explicit block barriers and writes only the final results.

It also selects 32-bit candidate positions before mapping back to 64-bit IDs.
The measured speedup is the combined effect of these changes, not an isolated
attribution to shared queries, fusion, or narrower selection payloads.

## Correctness and limitations

- All returned distances were finite, sorted, and agreed with baseline ranked
  distances within `2e-5 * max(1, abs(reference))`; IDs were in range.
- Up to 32 evenly spaced queries per case were checked against an independent
  FP64 CPU reference, including selected-ID distance, membership, and uniqueness.
- Exact ID sequences are not identical: the primary real-data case had 13,729
  different ID positions out of 64,000,000 (about 0.0215%). Ranked distances pass
  the tolerance checks; ties and different accumulation/merge ordering permit
  different IDs/order. This experiment does not prove bitwise equivalence.
- Final CUDA memcheck: **0 errors**. Final racecheck: **0 errors, 0 warnings**.
  Checks exercise odd dimensions, invalid candidate IDs, partial tiles, and k=256.
- The first prototype's reused RAFT block merge generated racecheck warnings.
  The final prototype uses an explicit barrier-protected tree merge. All results
  above were collected after that change; the initial sweep is exploratory only.
- The prototype supports FP32 squared L2, queries that are dataset rows, k <= 256,
  and at least k valid candidates. It is not installed into production cuVS.
- No end-to-end build or recall benchmark was run. Actual graph-build benefit
  depends on how much time the already optimized pipeline spends in refinement.

## Artifacts

- [Prototype](fused_refine.cuh), [benchmark](bench.cu), [unchanged baseline](baseline_refine.cuh)
- [Confirmed results, all variants](confirmation/summary.csv)
- [Environment and source/executable hashes](confirmation/metadata.json)
- [Memcheck log](confirmation/memcheck.stdout), [racecheck log](confirmation/racecheck.stdout)
- [Methodology and reproduction](README.md)

Per-case commands, CPU validation output, individual CUDA-event and wall-clock
samples, and aggregate timings are retained under `confirmation/`.
