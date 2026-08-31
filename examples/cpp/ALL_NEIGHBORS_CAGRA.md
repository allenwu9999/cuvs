# Multi-GPU all-neighbors CAGRA graph build

`MULTI_GPU_ALL_NEIGHBORS_CAGRA_EXAMPLE` constructs one global CAGRA adjacency graph from a large
FP32 dataset. The implementation is intended to make the multi-GPU all-neighbors workflow explicit
and reproducible; it is not a replacement for the higher-level CAGRA index API.

The design follows the all-neighbors path added to Faiss in
[facebookresearch/faiss#5282](https://github.com/facebookresearch/faiss/pull/5282): distribute
overlapping clusters across a RAFT single-node multi-GPU resource group, build local kNN graphs,
merge them using global vector IDs, and then prune the global kNN graph into a CAGRA graph. This
example uses the newer cuVS host-output overload rather than Faiss's device-output allocation. That
choice is important for degrees such as `64`: a `100M x 65` `int64` raw graph alone is about
48.43 GiB and cannot reside on one L40S.

## What the pipeline does

```text
FP32 dataset mapped from disk (one global row-ID space)
                         |
                         v
balanced clustering + top overlap_factor assignments
                         |
             +-----------+-----------+
             |           |           |
          GPU 0        GPU 1       GPU ...
       local clusters local clusters local clusters
       local IVF-PQ   local IVF-PQ   local IVF-PQ
             |           |           |
             +-----------+-----------+
                         |
          remap local neighbor IDs to global row IDs
                         |
     top-k merge into one host/file-backed global kNN graph
                         |
       remove self/invalid/duplicate IDs; int64 -> uint32
                         |
       public cagra::helpers::optimize (one selected GPU)
                         |
                         v
          one row-major uint32 [rows x graph_degree] graph
```

There are no independent final shards in this workflow. Overlapping cluster results are merged
during all-neighbors construction, so no portal selection or post-hoc graph stitching is needed.
Each row in the final file uses the original dataset's global row-ID space.

The executable asks all-neighbors for `intermediate_degree + 1` candidates. It then removes the
self-edge and any invalid or repeated IDs before giving exactly `intermediate_degree` candidates to
CAGRA pruning. A run fails rather than silently accepting a row that cannot be filled.

## Build

Build the current cuVS library and examples from the repository root:

```bash
./build.sh libcuvs
PARALLEL_LEVEL=8 ./examples/build.sh
```

The executable is created at:

```text
examples/cpp/build/MULTI_GPU_ALL_NEIGHBORS_CAGRA_EXAMPLE
```

## Input formats

All-neighbors currently accepts FP32 input. The example supports two row-major formats:

- `raw`: exactly `rows * dim` FP32 values and no header. Both `--rows` and `--dim` are required.
- `fbin`: two little-endian `uint32` values (`rows`, `dim`) followed by the FP32 values. Explicit
  `--rows` or `--dim` values are optional, but are validated when supplied.

The input is mapped read-only. It is not copied into one permanent device allocation; cuVS gathers
the rows for each overlapping cluster and transfers the cluster working set to its assigned GPU.

## Recommended 100M command on four L40S GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./examples/cpp/build/MULTI_GPU_ALL_NEIGHBORS_CAGRA_EXAMPLE \
  --dataset /fast-data/proxy_100m_128.f32 \
  --dataset-format raw \
  --rows 100000000 \
  --dim 128 \
  --output-dir /fast-data/all-neighbors-cagra-100m \
  --gpu-ids 0,1,2,3 \
  --build-algo ivf_pq \
  --clusters 64 \
  --overlap 2 \
  --intermediate-degree 64 \
  --graph-degree 32 \
  --refinement-rate 1.0 \
  --ivf-pq-search-batch 8192
```

The `8192` IVF-PQ internal search batch follows the large-scale Faiss implementation: it limits
temporary search workspace without changing the candidate results. The raw cuVS shape heuristic
uses `131072`, which can add a large workspace at 100M scale. `refinement-rate=1.0` avoids a second
candidate multiplier when build throughput, rather than proxy-dataset recall, is the experiment's
primary measurement.

To test NN-descent instead, replace the algorithm options with:

```bash
--build-algo nn_descent --nn-descent-iterations 20
```

## Output files

The successful default run retains only:

- `cagra_k32.u32`: the final row-major graph. Row `i` contains 32 global `uint32` neighbor IDs.
- `run-manifest.txt`: parameters, phase timings, conversion counters, layout, and semantic flags.

Temporary files are removed as soon as the next phase no longer needs them. Add
`--keep-intermediates` to retain:

- `all_neighbors_k65.i64`: raw all-neighbors IDs, including the extra self-edge allowance.
- `all_neighbors_k65.f32`: distances used while merging overlapping cluster results.
- `knn_k64.u32`: validated CAGRA optimizer input.

The final graph is **graph-only**. It is not a serialized CAGRA index and is not search-ready by
itself. CAGRA search also needs a compatible dataset attached in device-accessible storage. This
example deliberately stops before dataset attachment because `100M x 128` FP32 data alone is
47.68 GiB, larger than one L40S. It also does not request CAGRA's optional connectivity guarantee;
`run-manifest.txt` records both limitations explicitly.

## Memory and disk model for 100M x 128, degree 64 -> 32

Binary sizes are:

| Object | Shape and type | Size |
|---|---:|---:|
| Input dataset | `100M x 128 x float32` | 47.68 GiB |
| Raw neighbor IDs | `100M x 65 x int64` | 48.43 GiB |
| Raw distances | `100M x 65 x float32` | 24.21 GiB |
| Validated kNN graph | `100M x 64 x uint32` | 23.84 GiB |
| Final CAGRA graph | `100M x 32 x uint32` | 11.92 GiB |

With default cleanup, the output directory's peak allocation is about 72.64 GiB during the
all-neighbors phase (`raw IDs + distances`). During conversion it is about 72.27 GiB
(`raw IDs + validated graph`), and during pruning it is about 35.76 GiB
(`validated graph + final graph`). Keep at least 100 GiB free in the output filesystem; more is
needed when `--keep-intermediates` is enabled. The input dataset occupies its own 47.68 GiB file.

With `clusters=64` and `overlap=2`, the mean cluster contains approximately:

```text
100M * 2 / 64 = 3.125M vector appearances
```

For IVF-PQ with degree 65 and refinement 1.0, the explicit per-cluster device buffers are roughly
6 GiB at the mean cluster size, before IVF-PQ index and CUDA workspace allocations. Real cluster
sizes are imbalanced, so the largest cluster—not the mean—sets peak device memory. Increasing
`--clusters` reduces per-GPU peak memory approximately inversely, but adds fixed per-cluster work
and can reduce cross-cluster graph quality. Increasing `--overlap` improves cross-cluster candidate
coverage while increasing work and memory approximately linearly.

The public CAGRA optimizer currently runs on `--optimize-gpu` and uploads the validated
`100M x 64` graph (23.84 GiB) plus its workspace. The other GPUs are idle during that final phase.
Faiss #5282 includes a separate custom multi-GPU detour-count optimizer; this example intentionally
uses the public cuVS optimizer so its output semantics track cuVS rather than carrying a private
optimizer implementation.

## Parameter interactions

| Parameter | Increasing it usually does | Main risk |
|---|---|---|
| `clusters` | Lowers per-cluster GPU memory | More setup and weaker local coverage |
| `overlap` | Improves cross-cluster candidates | Nearly linear build/memory increase |
| `intermediate-degree` | Gives pruning more candidates | Larger host files and optimizer GPU peak |
| `graph-degree` | Produces a denser final graph | Larger graph and search cost |
| `refinement-rate` | Refines more IVF-PQ candidates exactly | Larger candidate buffers and longer build |
| `ivf-pq-search-batch` | Raises IVF-PQ throughput up to saturation | Larger temporary GPU workspace |
| `nn-descent-iterations` | Improves NN-descent convergence | Longer local builds |

Use at least as many clusters as GPUs. For balanced scheduling, choose a cluster count divisible by
the GPU count. The all-neighbors implementation assigns contiguous groups of clusters to ranks and
protects concurrent host updates with striped row locks. A vector assigned to two clusters can be
processed by different GPUs; the row lock makes the merge a safe read-modify-write operation.

## What is and is not measured

The manifest reports separate wall times for:

1. all-neighbors clustering, local graph builds, and global candidate merging;
2. self-edge removal, validation, and `int64` to `uint32` conversion;
3. CAGRA graph optimization;
4. total pipeline time, including file allocation and synchronous output flushes.

The run does not measure dataset generation, dataset attachment, CAGRA index serialization, or
search recall/QPS. Use a real dataset and an end-to-end search evaluation before interpreting the
proxy build time as a production result.
