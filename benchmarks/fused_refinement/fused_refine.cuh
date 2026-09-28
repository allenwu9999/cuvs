#pragma once
#include <raft/matrix/detail/select_warpsort.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace fusion_experiment {
namespace ws = raft::matrix::detail::select::warpsort;

// One CTA per query. Teams read contiguous coordinates of an original vector.
// Query coordinates and a 128-distance tile stay in shared memory. Top-k stays
// in RAFT's warp queues, merged at the end: no global candidate-distance matrix.
template <int Team, int Capacity, int Block = 128>
__global__ void fused_kernel(const float* dataset,
                             const int64_t* candidates,
                             int64_t* neighbors,
                             float* distances,
                             int64_t n_rows,
                             int dim,
                             int candidate_k,
                             int k,
                             int query_offset)
{
  extern __shared__ __align__(256) unsigned char storage[];
  float* query = reinterpret_cast<float*>(storage);
  float* tile = query + dim;
  const int tid = threadIdx.x;
  const int64_t row = int64_t(blockIdx.x) + query_offset;
  const int team = tid / Team;
  const int lane = tid % Team;
  constexpr int nteams = Block / Team;
  auto row_candidates = candidates + row * candidate_k;
  for (int d = tid; d < dim; d += Block) query[d] = dataset[row * dim + d];
  __syncthreads();

  using queue_t = ws::warp_sort_filtered<Capacity, true, float, uint32_t>;
  queue_t queue(k);
  for (int base = 0; base < candidate_k; base += Block) {
    #pragma unroll
    for (int group = 0; group < Team; ++group) {
      int j = base + group * nteams + team;
      int64_t id = j < candidate_k ? row_candidates[j] : -1;
      float acc = 0;
      if (id >= 0 && id < n_rows) {
        for (int d = lane; d < dim; d += Team) {
          float delta = query[d] - dataset[id * int64_t(dim) + d];
          acc += delta * delta;
        }
      } else {
        acc = CUDART_INF_F;
      }
      #pragma unroll
      for (int delta = Team / 2; delta > 0; delta /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, delta, Team);
      if (lane == 0) tile[group * nteams + team] = acc;
    }
    __syncthreads();
    queue.add(tile[tid], uint32_t(base + tid));
    __syncthreads();
  }
  // Query/tile storage can now be reused as the queue merge workspace.
  queue.done();
  float* partial_distances = reinterpret_cast<float*>(storage);
  uint32_t* partial_positions = reinterpret_cast<uint32_t*>(partial_distances + (Block / 64) * k);
  const int warp = tid / 32;
  // Explicit barriers on both sides of each merge avoid relying on warp
  // convergence as a memory fence when reusing shared queue storage.
  for (int width = 2; width <= Block / 32; width *= 2) {
    int slot = warp / width;
    if (warp % width == width / 2)
      queue.store(partial_distances + slot * k, partial_positions + slot * k);
    __syncthreads();
    if (warp % width == 0)
      queue.load_sorted(partial_distances + slot * k, partial_positions + slot * k);
    __syncthreads();
  }
  if (warp == 0) queue.store(distances + row * k, neighbors + row * k, raft::identity_op{},
              [=] __device__(uint32_t pos) -> int64_t {
                return pos < uint32_t(candidate_k) ? row_candidates[pos] : -1;
              });
}

template <int Team, int Capacity>
void launch(const float* data, const int64_t* candidates, int64_t* ids, float* dist,
            int64_t n, int b, int d, int c, int k, cudaStream_t stream, int offset = 0)
{
  const size_t smem = std::max(size_t(d + 128) * sizeof(float),
    size_t(ws::calc_smem_size_for_block_wide<float, uint32_t>(4, k)));
  fused_kernel<Team, Capacity><<<b, 128, smem, stream>>>(data, candidates, ids, dist, n, d, c, k, offset);
}

template <int Team>
void dispatch(const float* data, const int64_t* candidates, int64_t* ids, float* dist,
              int64_t n, int b, int d, int c, int k, cudaStream_t stream, int offset = 0)
{
  if (k <= 32) launch<Team, 32>(data, candidates, ids, dist, n, b, d, c, k, stream, offset);
  else if (k <= 64) launch<Team, 64>(data, candidates, ids, dist, n, b, d, c, k, stream, offset);
  else if (k <= 128) launch<Team, 128>(data, candidates, ids, dist, n, b, d, c, k, stream, offset);
  else if (k <= 256) launch<Team, 256>(data, candidates, ids, dist, n, b, d, c, k, stream, offset);
  else throw std::runtime_error("Experimental fused path supports k <= 256");
}
} // namespace fusion_experiment
