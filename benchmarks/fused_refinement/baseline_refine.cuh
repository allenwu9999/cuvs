/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/selection/select_k.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <limits>

namespace cuvs::neighbors::all_neighbors::detail {

// One warp evaluates one candidate directly from the resident cluster dataset.
// This avoids materializing a separate IVF list of candidate vectors per query.
template <typename T, typename IdxT>
RAFT_KERNEL candidate_l2_distances(const T* dataset,
                                   const IdxT* candidates,
                                   T* distances,
                                   IdxT n_rows,
                                   IdxT dim,
                                   IdxT query_offset,
                                   size_t n_pairs,
                                   size_t candidate_k)
{
  const size_t pair = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane    = threadIdx.x % 32;
  if (pair >= n_pairs) { return; }
  const IdxT query    = query_offset + pair / candidate_k;
  const IdxT neighbor = candidates[pair];
  T distance          = T{0};
  if (neighbor >= 0 && neighbor < n_rows) {
    for (IdxT col = lane; col < dim; col += 32) {
      const T diff = dataset[size_t(query) * dim + col] - dataset[size_t(neighbor) * dim + col];
      distance += diff * diff;
    }
  } else {
    distance = std::numeric_limits<T>::infinity();
  }
  for (int delta = 16; delta > 0; delta /= 2) {
    distance += __shfl_down_sync(0xffffffff, distance, delta);
  }
  if (lane == 0) { distances[pair] = distance; }
}

// All-neighbors IVF-PQ supports squared L2. Reuse its approximate-distance
// buffer for exact candidate distances and select sorted results on device.
template <typename T, typename IdxT>
void refine_candidates(raft::resources const& res,
                       raft::device_matrix_view<const T, IdxT> dataset,
                       raft::device_matrix_view<const IdxT, IdxT> candidates,
                       raft::device_matrix_view<T, IdxT> candidate_distances,
                       raft::device_matrix_view<IdxT, IdxT> neighbors,
                       raft::device_matrix_view<T, IdxT> distances)
{
  const auto n_rows      = dataset.extent(0);
  const auto dim         = dataset.extent(1);
  const auto candidate_k = candidates.extent(1);
  const auto k           = neighbors.extent(1);
  // Bound selection workspace independently of the cluster size.
  constexpr IdxT query_batch_size = 8192;
  constexpr int block_size        = 256;
  for (IdxT offset = 0; offset < n_rows; offset += query_batch_size) {
    const auto rows       = std::min(query_batch_size, n_rows - offset);
    const size_t pairs    = size_t(rows) * candidate_k;
    auto batch_candidates = raft::make_device_matrix_view<const IdxT, IdxT>(
      candidates.data_handle() + size_t(offset) * candidate_k, rows, candidate_k);
    auto batch_distances = raft::make_device_matrix_view<T, IdxT>(
      candidate_distances.data_handle() + size_t(offset) * candidate_k, rows, candidate_k);
    candidate_l2_distances<<<raft::ceildiv(pairs, size_t(block_size / 32)),
                             block_size,
                             0,
                             raft::resource::get_cuda_stream(res)>>>(dataset.data_handle(),
                                                                     batch_candidates.data_handle(),
                                                                     batch_distances.data_handle(),
                                                                     n_rows,
                                                                     dim,
                                                                     offset,
                                                                     pairs,
                                                                     candidate_k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    cuvs::selection::select_k(
      res,
      raft::make_const_mdspan(batch_distances),
      std::make_optional(batch_candidates),
      raft::make_device_matrix_view<T, IdxT>(distances.data_handle() + size_t(offset) * k, rows, k),
      raft::make_device_matrix_view<IdxT, IdxT>(
        neighbors.data_handle() + size_t(offset) * k, rows, k),
      true,
      true);
  }
}

}  // namespace cuvs::neighbors::all_neighbors::detail
