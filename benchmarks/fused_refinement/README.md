# Fused direct-access refinement experiment

This standalone experiment compares the GPU refinement from cuVS commit
`2fbbb6d404c4644f591e180907f2f845e3733af8` with a prototype that shares queries and
fuses exact squared-L2 distance computation with top-k selection. Production cuVS
sources are not modified.

## Implementation

- `baseline_refine.cuh` is an unchanged copy of the commit's refinement header.
  When all dataset rows are queried, the harness calls that original function
  directly. For query prefixes, it uses the same original distance kernel,
  8192-row batching, and `cuvs::selection::select_k` invocation.
- `fused_refine.cuh` uses one 128-thread CTA per query. A team of 4, 8, 16, or 32
  threads reads consecutive coordinates of each candidate from the resident,
  row-major original dataset. Query coordinates are shared within the CTA.
- A 128-element distance tile remains in shared memory. RAFT warp top-k queues
  retain candidates locally; an explicit, barrier-protected tree merge combines
  the warp queues. Only the final IDs and distances are written to global memory.
- Selection uses 32-bit candidate positions, then maps them back to 64-bit IDs.
- The prototype accepts FP32, squared L2, dataset rows as queries, and `k <= 256`.
  It assumes finite input coordinates and at least k valid candidates per row.
  It is not a general replacement for the public refine API.
- Methods 108/116/132 additionally limit each fused launch to 8192 queries,
  matching the baseline's batching. Methods 8/16/32 launch all query CTAs at once.

## Benchmark protocol

The machine has one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), not the
L40S hardware from the earlier graph-build experiment. The image used is
`kuaishou-cuvs-all-neighbors-fp16:26.08`, ID
`sha256:b803154f20d6059a6cd367b1aa204b39f70b2a26fa07b0619aa100b983949f3c`.
Compilation uses CUDA 12.9, `-O3 -lineinfo -arch=sm_120`, without fast-math.

All compared methods share the same data and precomputed candidate IDs within
each process. Allocations, data initialization/loading, IVF-PQ build/search, and
correctness checks are outside the timer. The RMM allocator uses a memory pool.
CUDA events measure repeated complete refinement calls (including selection).
Host synchronized wall time is also recorded. These measurements include host
submission gaps visible on the CUDA stream; CUDA graph capture is not used.
Each method receives warm-up calls, and method order is shuffled deterministically
between rounds. No other benchmark is launched concurrently by this experiment.

Synthetic candidates are unique, scattered IDs. The real-data case uses the
first 1,000,000 vectors of Deep Image (96 dimensions) and actual IVF-PQ candidates:
500 lists, 16 probes, PQ dimension 64, 4 bits, 10 k-means iterations, training
fraction 0.1, FP32 search arithmetic, and search batch size 8192. Search and build
are performed once per case, outside timing. A refinement ratio of 2 means
128 candidates -> 64 output neighbors, or 256 -> 128 in the larger-degree case.

## Validation and artifacts

Every output is checked for finite, sorted distances, valid IDs, and agreement
with baseline ranked distances within `2e-5 * max(1, abs(reference))`. For up to
32 deterministic query rows spread across the workload, all selected values and
IDs are additionally checked against independently computed FP64 candidate
distances, candidate membership, and uniqueness. Exact ID equality is recorded,
but is not required because ties and accumulation order can change ordering.

An initial tuning sweep identified useful thread-team sizes. A subsequent race
check identified shared-memory synchronization warnings in the reused block queue
merge. The current prototype uses explicit barriers around the merge; authoritative
results and sanitizer logs are in `confirmation/`. The earlier exploratory code
and timings are omitted from this package. The three CUDA source files are
preserved byte-for-byte as measured; their hashes match `confirmation/metadata.json`.

CUDA Compute Sanitizer 12.9.79 was downloaded from NVIDIA's official CUDA 12.9.1
redistribution manifest and its SHA-256 verified. Download it with
`download_sanitizer.sh`; downloaded binaries are excluded from version control.

## Reproduce

From the repository root on the host, set `DEEP_DIR` to a directory containing the
Deep Image `base.fbin` file (two int32 header values: rows and dimensions, followed
by row-major FP32 coordinates). The recorded dataset has 9,990,000 rows and 96
dimensions; the benchmark reads the first million rows.

The named container image is a locally available development image, not a public
image distributed by this experiment. It contains CUDA 12.9, cuVS, RAFT, and RMM
headers/libraries under `/opt/conda`. Reproduction on another machine requires
that image or a compatible development environment. The build targets `sm_120`;
change it for a different GPU, and record the resulting environment separately.

```sh
BENCH_DIR="$(pwd)/benchmarks/fused_refinement"
DEEP_DIR=/path/to/deep-image-96-inner
bash "$BENCH_DIR/download_sanitizer.sh"

docker run --rm --gpus all \
  -v "$BENCH_DIR:/bench" \
  --entrypoint /bin/bash kuaishou-cuvs-all-neighbors-fp16:26.08 /bench/build.sh

docker run --rm --gpus all \
  -v "$BENCH_DIR:/bench" \
  -v "$DEEP_DIR:/datasets/deep:ro" \
  --entrypoint python kuaishou-cuvs-all-neighbors-fp16:26.08 /bench/confirm.py
```

The confirmation script runs sanitizer checks before performance measurements,
and records exact commands, source/executable hashes, GPU metadata, individual
rounds, medians/ranges, and validation output. New runs go to `runs/confirmation/`
(ignored by Git), preserving the committed results under `confirmation/`.
Re-running overwrites `runs/confirmation/`; move that directory aside to preserve
multiple runs. The original measured executable is excluded from version control;
its hash is retained as provenance, not a promise of reproducible binary output.
