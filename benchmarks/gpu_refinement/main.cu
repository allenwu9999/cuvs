#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <vector>

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void sync_all(int gpus)
{
  for (int i = 0; i < gpus; ++i) {
    RAFT_CUDA_TRY(cudaSetDevice(i));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
  }
  RAFT_CUDA_TRY(cudaSetDevice(0));
}
int main(int argc, char** argv)
try {
  if (argc < 5)
    throw std::runtime_error(
      "usage: bench base.fbin gpus rows(0=all) output_prefix [clusters=16] [k=64] "
      "[host|device|device-no-dist|device-input=host] [device|managed= device]");
  const int gpus = std::stoi(argv[2]);
  std::ifstream input(argv[1], std::ios::binary);
  uint32_t shape[2];
  input.read(reinterpret_cast<char*>(shape), 8);
  int64_t n = std::stoll(argv[3]), dim = shape[1];
  if (!n) n = shape[0];
  if (!input || n > shape[0] || dim < 1) throw std::runtime_error("invalid input");
  int64_t k                   = argc > 6 ? std::stoll(argv[6]) : 64;
  int clusters                = argc > 5 ? std::stoi(argv[5]) : 16;
  std::string mode            = argc > 7 ? argv[7] : "host";
  std::string optimize_memory = argc > 8 ? argv[8] : "device";
  if (optimize_memory != "device" && optimize_memory != "managed")
    throw std::runtime_error("optimizer memory must be device or managed");
  std::vector<float> data(n * dim);
  if (std::string(argv[1]).ends_with(".u8bin")) {
    // Exact uint8 -> FP32 conversion; no scaling or synthesized vectors.
    constexpr size_t chunk_size = 128 * 1024 * 1024;
    std::vector<uint8_t> chunk(chunk_size);
    for (size_t offset = 0; offset < data.size(); offset += chunk_size) {
      size_t count = std::min(chunk_size, data.size() - offset);
      input.read(reinterpret_cast<char*>(chunk.data()), count);
      if (!input) throw std::runtime_error("short uint8 input");
#pragma omp parallel for
      for (size_t i = 0; i < count; ++i)
        data[offset + i] = static_cast<float>(chunk[i]);
    }
  } else {
    input.read(reinterpret_cast<char*>(data.data()), data.size() * 4);
    if (!input) throw std::runtime_error("short input");
  }
  std::vector<int> devices(gpus);
  std::iota(devices.begin(), devices.end(), 0);
  raft::device_resources_snmg res(devices);
  res.set_memory_pool(70);
  cuvs::neighbors::all_neighbors::all_neighbors_params params;
  params.n_clusters     = clusters;
  params.overlap_factor = 2;
  params.metric         = cuvs::distance::DistanceType::L2Expanded;
  int64_t cluster_rows  = clusters == 1 ? n : n * 2 / clusters;
  cuvs::neighbors::graph_build_params::ivf_pq_params pq(
    raft::matrix_extent<int64_t>{cluster_rows, dim});
  pq.refinement_rate = 2.0;
  // Raw SIFT squared distances exceed FP16 range. Use FP32 search arithmetic
  // for both datasets and both refinement implementations.
  pq.search_params.lut_dtype               = CUDA_R_32F;
  pq.search_params.internal_distance_dtype = CUDA_R_32F;
  pq.search_params.coarse_search_dtype     = CUDA_R_32F;
  pq.build_params.n_lists                  = std::max<uint32_t>(16, cluster_rows / 2000);
  pq.search_params.n_probes                = std::min<uint32_t>(pq.build_params.n_lists, 16);
  pq.search_params.max_internal_batch_size = 8192;
  params.graph_build_params                = pq;
  std::vector<int64_t> neighbors(n * k);
  std::vector<float> distances(n * k);
  auto dataset = raft::make_host_matrix_view<const float, int64_t>(data.data(), n, dim);
  auto nh      = raft::make_host_matrix_view<int64_t, int64_t>(neighbors.data(), n, k);
  auto dh      = raft::make_host_matrix_view<float, int64_t>(distances.data(), n, k);
  // Warm library kernels outside timed runs, using a small real-data prefix.
  if (n > 20000) {
    constexpr int64_t warm_rows    = 20000;
    auto warm_params               = params;
    warm_params.n_clusters         = 4;
    auto warm_pq                   = pq;
    warm_pq.build_params.n_lists   = 16;
    warm_pq.search_params.n_probes = 16;
    warm_params.graph_build_params = warm_pq;
    std::vector<int64_t> warm_ids(warm_rows * k);
    std::vector<float> warm_dist(warm_rows * k);
    cuvs::neighbors::all_neighbors::build(
      res,
      warm_params,
      raft::make_host_matrix_view<const float, int64_t>(data.data(), warm_rows, dim),
      raft::make_host_matrix_view<int64_t, int64_t>(warm_ids.data(), warm_rows, k),
      raft::make_host_matrix_view<float, int64_t>(warm_dist.data(), warm_rows, k));
    std::vector<uint32_t> warm_graph(warm_ids.begin(), warm_ids.end()),
      warm_out(warm_rows * (k / 2));
    cuvs::neighbors::cagra::helpers::optimize(
      res,
      raft::make_host_matrix_view<uint32_t, int64_t>(warm_graph.data(), warm_rows, k),
      raft::make_host_matrix_view<uint32_t, int64_t>(warm_out.data(), warm_rows, k / 2));
  }
  // Prime CUDA runtime on every selected device outside the build timer.
  sync_all(gpus);
  auto start = Clock::now();
  if (mode == "host") {
    cuvs::neighbors::all_neighbors::build(res, params, dataset, nh, dh);
  } else {
    auto nd = raft::make_device_matrix<int64_t, int64_t>(res, n, k);
    auto dd = raft::make_device_matrix<float, int64_t>(res, n, k);
    std::optional<raft::device_matrix_view<float, int64_t>> dv = dd.view();
    if (mode == "device-no-dist") dv = std::nullopt;
    if (mode == "device-input") {
      auto xd = raft::make_device_matrix<float, int64_t>(res, n, dim);
      raft::update_device(
        xd.data_handle(), data.data(), data.size(), raft::resource::get_cuda_stream(res));
      cuvs::neighbors::all_neighbors::build(
        res, params, raft::make_const_mdspan(xd.view()), nd.view(), dv);
    } else
      cuvs::neighbors::all_neighbors::build(res, params, dataset, nd.view(), dv);
    raft::update_host(
      neighbors.data(), nd.data_handle(), neighbors.size(), raft::resource::get_cuda_stream(res));
    if (dv)
      raft::update_host(
        distances.data(), dd.data_handle(), distances.size(), raft::resource::get_cuda_stream(res));
    sync_all(gpus);
  }
  sync_all(gpus);
  double build = elapsed(start);
  std::cout << "BUILD_SECONDS " << build << std::endl;
  auto convert_start = Clock::now();
  std::vector<uint32_t> graph(neighbors.size()), optimized(n * (k / 2));
#pragma omp parallel for
  for (size_t i = 0; i < neighbors.size(); ++i)
    graph[i] = static_cast<uint32_t>(neighbors[i]);
  double convert = elapsed(convert_start);
  // For graphs larger than device memory, only the optimizer's large workspace
  // uses managed memory. The all-neighbors allocator and timers are unchanged.
  auto opt_start = Clock::now();
  raft::device_resources optimizer_res;
  if (optimize_memory == "managed") {
    // Release the now-unused builder pools so managed optimizer pages can use
    // the GPU's full memory capacity. Include this setup in optimizer time.
    res.set_memory_pool(0);
    raft::resource::set_large_workspace_resource(
      optimizer_res, raft::mr::device_resource{rmm::mr::managed_memory_resource{}});
  }
  const raft::resources& optimizer_handle =
    optimize_memory == "managed" ? static_cast<const raft::resources&>(optimizer_res) : res;
  cuvs::neighbors::cagra::helpers::optimize(
    optimizer_handle,
    raft::make_host_matrix_view<uint32_t, int64_t>(graph.data(), n, k),
    raft::make_host_matrix_view<uint32_t, int64_t>(optimized.data(), n, k / 2));
  sync_all(gpus);
  double optimize = elapsed(opt_start);
  std::cout << "OPTIMIZE_SECONDS " << optimize << std::endl;
  // Structural validation is deliberately outside every reported build timer.
  uint64_t invalid = 0, duplicates = 0, unsorted = 0;
#pragma omp parallel for reduction(+ : invalid, duplicates, unsorted)
  for (int64_t i = 0; i < n; ++i) {
    std::vector<int64_t> row(neighbors.begin() + i * k, neighbors.begin() + (i + 1) * k);
    for (int64_t j = 0; j < k; ++j) {
      if (row[j] < 0 || row[j] >= n ||
          (mode != "device-no-dist" && !std::isfinite(distances[i * k + j])))
        ++invalid;
      if (j && mode != "device-no-dist" && distances[i * k + j] + 1e-4f < distances[i * k + j - 1])
        ++unsorted;
    }
    std::sort(row.begin(), row.end());
    for (int64_t j = 1; j < k; ++j)
      if (row[j] == row[j - 1]) ++duplicates;
    std::vector<uint32_t> out(optimized.begin() + i * (k / 2),
                              optimized.begin() + (i + 1) * (k / 2));
    for (auto id : out)
      if (id >= n) ++invalid;
    std::sort(out.begin(), out.end());
    for (size_t j = 1; j < out.size(); ++j)
      if (out[j] == out[j - 1]) ++duplicates;
  }
  // Deterministically sample database rows for exact-distance/recall checks.
  std::ofstream sample(std::string(argv[4]) + ".samples", std::ios::binary);
  for (int64_t s = 0; s < 256; ++s) {
    int64_t row = (s * n / 256 + 17) % n;
    sample.write(reinterpret_cast<char*>(&row), 8);
    sample.write(reinterpret_cast<char*>(neighbors.data() + row * k), k * 8);
    sample.write(reinterpret_cast<char*>(distances.data() + row * k), k * 4);
  }
  std::ofstream result(std::string(argv[4]) + ".json");
  result << std::setprecision(9) << "{\"rows\":" << n << ",\"dim\":" << dim << ",\"gpus\":" << gpus
         << ",\"clusters\":" << clusters << ",\"k\":" << k
         << ",\"n_lists\":" << pq.build_params.n_lists
         << ",\"n_probes\":" << pq.search_params.n_probes
         << ",\"pq_dim\":" << pq.build_params.pq_dim << ",\"pq_bits\":" << pq.build_params.pq_bits
         << ",\"optimizer_memory\":\"" << optimize_memory << "\""
         << ",\"refinement_rate\":2,\"build_seconds\":" << build
         << ",\"convert_seconds\":" << convert << ",\"optimize_seconds\":" << optimize
         << ",\"total_seconds\":" << (build + convert + optimize) << ",\"invalid\":" << invalid
         << ",\"duplicates\":" << duplicates << ",\"unsorted\":" << unsorted << "}\n";
  std::cout << "VALIDATION invalid=" << invalid << " duplicates=" << duplicates
            << " unsorted=" << unsorted << std::endl;
  return invalid || duplicates || unsorted ? 2 : 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
