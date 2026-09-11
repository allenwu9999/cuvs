#include "../../cpp/src/neighbors/all_neighbors/all_neighbors_refine.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <random>
#include <tuple>
#include <vector>

int main()
try {
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  // Cross the 8192-query boundary, include an odd dimension and a degree >256.
  for (auto [dim, candidate_k, k] :
       {std::tuple<int64_t, int64_t, int64_t>{137, 47, 23}, {33, 603, 300}}) {
    constexpr int64_t n = 8209;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<float> x(n * dim);
    for (auto& value : x)
      value = uniform(rng);
    std::vector<int64_t> candidates(n * candidate_k);
    for (int64_t row = 0; row < n; ++row) {
      for (int64_t j = 0; j < candidate_k; ++j)
        candidates[row * candidate_k + j] = (row * 17 + j * 37) % n;
      candidates[row * candidate_k + candidate_k - 1] = -1;
      candidates[row * candidate_k + candidate_k - 2] = n;
    }
    auto xd        = raft::make_device_matrix<float, int64_t>(res, n, dim);
    auto cd        = raft::make_device_matrix<int64_t, int64_t>(res, n, candidate_k);
    auto scratch   = raft::make_device_matrix<float, int64_t>(res, n, candidate_k);
    auto ids       = raft::make_device_matrix<int64_t, int64_t>(res, n, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n, k);
    raft::update_device(xd.data_handle(), x.data(), x.size(), stream);
    raft::update_device(cd.data_handle(), candidates.data(), candidates.size(), stream);
    cuvs::neighbors::all_neighbors::detail::refine_candidates(res,
                                                              raft::make_const_mdspan(xd.view()),
                                                              raft::make_const_mdspan(cd.view()),
                                                              scratch.view(),
                                                              ids.view(),
                                                              distances.view());
    std::vector<int64_t> result_ids(n * k);
    std::vector<float> result_distances(n * k);
    raft::update_host(result_ids.data(), ids.data_handle(), result_ids.size(), stream);
    raft::update_host(
      result_distances.data(), distances.data_handle(), result_distances.size(), stream);
    raft::resource::sync_stream(res);
    uint64_t failures = 0;
#pragma omp parallel for reduction(+ : failures)
    for (int64_t row = 0; row < n; ++row) {
      auto distance = [&](int64_t id) {
        double sum = 0;
        for (int64_t col = 0; col < dim; ++col) {
          double diff = double(x[row * dim + col]) - double(x[id * dim + col]);
          sum += diff * diff;
        }
        return sum;
      };
      std::vector<double> exact;
      for (int64_t j = 0; j < candidate_k - 2; ++j)
        exact.push_back(distance(candidates[row * candidate_k + j]));
      std::sort(exact.begin(), exact.end());
      std::vector<int64_t> selected;
      for (int64_t j = 0; j < k; ++j) {
        const auto id   = result_ids[row * k + j];
        const auto dist = result_distances[row * k + j];
        if (id < 0 || id >= n || !std::isfinite(dist)) {
          ++failures;
          continue;
        }
        if (std::abs(dist - exact[j]) > 1e-4 || std::abs(dist - distance(id)) > 1e-4) ++failures;
        if (std::find(candidates.begin() + row * candidate_k,
                      candidates.begin() + (row + 1) * candidate_k,
                      id) == candidates.begin() + (row + 1) * candidate_k)
          ++failures;
        selected.push_back(id);
      }
      std::sort(selected.begin(), selected.end());
      if (std::adjacent_find(selected.begin(), selected.end()) != selected.end()) ++failures;
    }
    std::cout << "dim=" << dim << " candidates=" << candidate_k << " k=" << k
              << " failures=" << failures << std::endl;
    if (failures) return 1;
  }
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
