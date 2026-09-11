# GPU refinement experiment

The production change is in `cpp/src/neighbors/all_neighbors/all_neighbors_builder.cuh`.
It refines IVF-PQ candidates on the GPU against the already resident cluster data,
in query batches of 8192. Full builds write straight into device output. Batched
builds stage only the refined k IDs for the existing remap/merge implementation.
It computes squared L2 directly with a warp per candidate, then uses cuVS
sorted top-k selection. It supports degrees above 256 without a CPU fallback. This change does not move the existing host graph merge to the GPU.

The two benchmark executables compile the original and modified all-neighbors
translation units and link the same installed cuVS library for IVF-PQ, refinement,
and CAGRA optimization. Baseline commit: `c740818bc68c58e5141c0eb2f832f26cfd9288bd`.
The baseline must be an unmodified checkout or archive of that commit.

## Local build and execution

The recorded experiment uses the existing local image
`kuaishou-cuvs-all-neighbors-fp16:26.08` (image ID
`sha256:850ab92b2b122a0cb25a953856e24edd82e2878e137281af447859fe9b1ae31f`), with `/tmp/cuvs` mounted at `/work`, the
baseline source at `/baseline`, and `/home/wua/scratch/cuvs-bench` mounted read-only
at `/datasets`. The benchmark operates on FP32 input. The image name does not
indicate the data type of this experiment.

From the host, prepare a baseline and start the benchmark environment:

```sh
mkdir -p /tmp/cuvs-refine-baseline
git -C /tmp/cuvs archive c740818bc68c58e5141c0eb2f832f26cfd9288bd | \
  tar -x -C /tmp/cuvs-refine-baseline
docker run --rm -it --gpus all \
  -v /tmp/cuvs:/work -v /tmp/cuvs-refine-baseline:/baseline:ro \
  -v /home/wua/scratch/cuvs-bench:/datasets:ro \
  --entrypoint /bin/bash kuaishou-cuvs-all-neighbors-fp16:26.08
```

Then run inside the container (move the existing `results/timings` and
`results/smoke` directories aside to collect a fresh run):

```sh
cmake -S /work/benchmarks/gpu_refinement -B /work/benchmarks/gpu_refinement/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH=/opt/conda -DCUVS_SOURCE=/work -DCUVS_BASELINE=/baseline
cmake --build /work/benchmarks/gpu_refinement/build -j 4
python -m cuvs_bench.get_dataset --dataset sift-128-euclidean \
  --dataset-path /work/benchmarks/gpu_refinement/datasets
export CUDA_PATH=/opt/conda/targets/x86_64-linux
python /work/benchmarks/gpu_refinement/run.py --smoke
python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/smoke
python /work/benchmarks/gpu_refinement/run.py --repeats 3
python /work/benchmarks/gpu_refinement/validate.py /work/benchmarks/gpu_refinement/results/timings
cmake --build /work/benchmarks/gpu_refinement/build --target refine_test -j 2
OMP_NUM_THREADS=16 /work/benchmarks/gpu_refinement/build/refine_test
python /work/benchmarks/gpu_refinement/summarize.py
```

CMake creates a private RMM header overlay with an inline `get()` alias for
`value()`: the source checkout uses the newer name, and this image has the older
name. Both executables use this identical, ABI-neutral compatibility shim.

## Measurement

- Four NVIDIA L40S (48 GB marketed capacity each), dual AMD EPYC 7742, 1 TiB RAM.
- The GPU count selects actual RAFT SNMG ranks and visible CUDA devices.
- Runs execute sequentially, alternating CPU/GPU order between repetitions.
- Fixed 64-thread CPU budget: `OMP_NUM_THREADS=64,64/gpus`, nested OpenMP enabled,
  passive waiting, dynamic teams disabled. The outer rank loop explicitly uses
  one thread per GPU. Top-level CPU loops may use 64 threads.
- Both datasets use FP32 vectors and squared L2. Deep Image comes from the
  normalized `deep-image-96-inner/base.fbin` cuVS Bench dataset. SIFT is downloaded
  and converted with the cuVS Bench dataset helper.
- 16 clusters, overlap factor 2, intermediate degree 64, refinement rate 2,
  CAGRA degree 32. IVF-PQ PQ dimension 64, 4 bits, 16 probes, 8192 search batch,
  10 k-means iterations. Number of lists is the nominal cluster size / 2000
  (minimum 16). Training fraction follows `ivf_pq_params`' shape-based defaults.
- IVF-PQ LUT, internal distances, and coarse search use FP32 for both datasets;
  raw SIFT distances overflow the default FP16 arithmetic.
- RMM reserves a 70% device-memory pool per selected GPU before timing.
- Input loading, output allocation, and a 20K-row warm-up (all-neighbors and
  CAGRA optimization) occur before timing. Allocation of intermediate builder
  buffers is included. Every build timer ends with synchronization of all GPUs.
- `build_seconds` measures complete all-neighbors graph construction, including
  clustering, IVF-PQ build/search, refinement, transfers, and host merge.
- `convert_seconds` includes allocation and conversion of the int64 graph to
  uint32 plus allocation of final CAGRA output. `optimize_seconds` measures the
  subsequent CAGRA optimization on GPU 0. `total_seconds` is their sum.
- These are graph-construction timings, not query throughput or complete CAGRA
  index serialization. Dataset attachment and disk serialization are not timed.
- Full-graph ID, uniqueness, finite-distance, and ordering checks follow timing.
  The final CAGRA graph is also checked for valid, unique IDs.
- Untimed `validate.py` recomputes exact top-k for 256 deterministic database
  queries and compares returned edge distances to FP64 direct calculations.
  Ground truth is computed for the actual all-neighbors queries, not taken from
  the benchmark dataset's separate held-out query set.

Raw per-run JSON, logs, and sampled edges are under `results/`. Generated binaries,
downloaded datasets, and raw runs are ignored by git. `run.py` resumes completed
runs; use a fresh results directory when changing the implementation or settings.

`refine_test` checks every result against a CPU FP64 candidate reference for
8,209 queries (crossing the 8,192-query batch boundary), dimensions 137 and 33,
47/603 candidates, k=23/300, and invalid candidate sentinels. Run this outside
the timing matrix so it does not compete for GPU resources.

## 100M-vector extension

[REPORT-100M.md](REPORT-100M.md) records the separate real BIGANN 100M × 128
experiment, with 64 clusters and one warmed observation for each CPU/GPU
refinement variant at 1, 2, and 4 GPUs. It uses exact uint8-to-FP32 conversion
and normal device-memory CAGRA pruning. Its dataset preparation, runner,
measurement records, and results are separate from the earlier smaller runs.
