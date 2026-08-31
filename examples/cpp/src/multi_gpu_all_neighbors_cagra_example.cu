/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
  std::filesystem::path dataset_path;
  std::filesystem::path output_dir;
  std::string dataset_format = "raw";
  std::string build_algo     = "ivf_pq";
  int64_t rows               = 0;
  int64_t dim                = 0;
  std::vector<int> gpu_ids;
  int optimize_gpu             = -1;
  size_t n_clusters            = 64;
  size_t overlap_factor        = 2;
  int64_t intermediate_degree  = 64;
  int64_t graph_degree         = 32;
  float refinement_rate        = 1.0f;
  uint32_t ivf_pq_search_batch = 8192;
  uint32_t ivf_pq_n_lists      = 0;
  size_t nn_descent_iterations = 20;
  bool keep_intermediates      = false;
};

[[noreturn]] void usage(const char* program, const std::string& error = {})
{
  if (!error.empty()) { std::cerr << "error: " << error << "\n\n"; }
  std::cerr
    << "Usage: " << program << " --dataset FILE --output-dir DIR [options]\n\n"
    << "Required:\n"
    << "  --dataset FILE                 FP32 row-major dataset\n"
    << "  --output-dir DIR               Work directory and final graph location\n\n"
    << "Dataset:\n"
    << "  --dataset-format raw|fbin      raw has no header; fbin starts with two uint32 values\n"
    << "  --rows N                       Required for raw; optional validation for fbin\n"
    << "  --dim D                        Required for raw; optional validation for fbin\n\n"
    << "GPU and graph:\n"
    << "  --gpu-ids 0,1,2,3             GPUs used by all-neighbors (default: all visible)\n"
    << "  --optimize-gpu ID              GPU used for CAGRA pruning (default: first GPU)\n"
    << "  --build-algo ivf_pq|nn_descent\n"
    << "  --clusters N                   Overlapping clusters (default: 64)\n"
    << "  --overlap N                    Clusters per vector (default: 2)\n"
    << "  --intermediate-degree N        kNN degree passed to CAGRA pruning (default: 64)\n"
    << "  --graph-degree N               Final CAGRA degree (default: 32)\n\n"
    << "IVF-PQ:\n"
    << "  --refinement-rate X            Exact-refinement multiplier (default: 1.0)\n"
    << "  --ivf-pq-search-batch N        IVF-PQ workspace cap; 0 uses cuVS default\n"
    << "  --ivf-pq-n-lists N             0 uses cuVS shape heuristic\n\n"
    << "NN-descent:\n"
    << "  --nn-descent-iterations N      Maximum iterations (default: 20)\n\n"
    << "Output:\n"
    << "  --keep-intermediates           Retain raw kNN IDs, distances, and converted graph\n"
    << "  --help                         Show this message\n";
  std::exit(error.empty() ? EXIT_SUCCESS : EXIT_FAILURE);
}

template <typename T>
T parse_number(const char* value, const char* name)
{
  std::istringstream input(value);
  T result{};
  input >> result;
  if (!input || !input.eof()) {
    usage("multi_gpu_all_neighbors_cagra_example",
          std::string("invalid value for ") + name + ": " + value);
  }
  return result;
}

std::vector<int> parse_gpu_ids(const std::string& value)
{
  std::vector<int> ids;
  std::stringstream input(value);
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token.empty()) { usage("multi_gpu_all_neighbors_cagra_example", "empty GPU id"); }
    ids.push_back(parse_number<int>(token.c_str(), "--gpu-ids"));
  }
  if (ids.empty()) { usage("multi_gpu_all_neighbors_cagra_example", "--gpu-ids is empty"); }
  auto sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    usage("multi_gpu_all_neighbors_cagra_example", "--gpu-ids contains a duplicate");
  }
  return ids;
}

options parse_options(int argc, char** argv)
{
  options opts;
  auto next = [&](int& i, const char* name) -> const char* {
    if (++i >= argc) { usage(argv[0], std::string("missing value for ") + name); }
    return argv[i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dataset") {
      opts.dataset_path = next(i, "--dataset");
    } else if (arg == "--output-dir") {
      opts.output_dir = next(i, "--output-dir");
    } else if (arg == "--dataset-format") {
      opts.dataset_format = next(i, "--dataset-format");
    } else if (arg == "--rows") {
      opts.rows = parse_number<int64_t>(next(i, "--rows"), "--rows");
    } else if (arg == "--dim") {
      opts.dim = parse_number<int64_t>(next(i, "--dim"), "--dim");
    } else if (arg == "--gpu-ids") {
      opts.gpu_ids = parse_gpu_ids(next(i, "--gpu-ids"));
    } else if (arg == "--optimize-gpu") {
      opts.optimize_gpu = parse_number<int>(next(i, "--optimize-gpu"), "--optimize-gpu");
    } else if (arg == "--build-algo") {
      opts.build_algo = next(i, "--build-algo");
    } else if (arg == "--clusters") {
      opts.n_clusters = parse_number<size_t>(next(i, "--clusters"), "--clusters");
    } else if (arg == "--overlap") {
      opts.overlap_factor = parse_number<size_t>(next(i, "--overlap"), "--overlap");
    } else if (arg == "--intermediate-degree") {
      opts.intermediate_degree =
        parse_number<int64_t>(next(i, "--intermediate-degree"), "--intermediate-degree");
    } else if (arg == "--graph-degree") {
      opts.graph_degree = parse_number<int64_t>(next(i, "--graph-degree"), "--graph-degree");
    } else if (arg == "--refinement-rate") {
      opts.refinement_rate = parse_number<float>(next(i, "--refinement-rate"), "--refinement-rate");
    } else if (arg == "--ivf-pq-search-batch") {
      opts.ivf_pq_search_batch =
        parse_number<uint32_t>(next(i, "--ivf-pq-search-batch"), "--ivf-pq-search-batch");
    } else if (arg == "--ivf-pq-n-lists") {
      opts.ivf_pq_n_lists = parse_number<uint32_t>(next(i, "--ivf-pq-n-lists"), "--ivf-pq-n-lists");
    } else if (arg == "--nn-descent-iterations") {
      opts.nn_descent_iterations =
        parse_number<size_t>(next(i, "--nn-descent-iterations"), "--nn-descent-iterations");
    } else if (arg == "--keep-intermediates") {
      opts.keep_intermediates = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0], "unknown argument: " + arg);
    }
  }

  if (opts.dataset_path.empty()) { usage(argv[0], "--dataset is required"); }
  if (opts.output_dir.empty()) { usage(argv[0], "--output-dir is required"); }
  if (opts.dataset_format != "raw" && opts.dataset_format != "fbin") {
    usage(argv[0], "--dataset-format must be raw or fbin");
  }
  if (opts.dataset_format == "raw" && (opts.rows <= 0 || opts.dim <= 0)) {
    usage(argv[0], "raw input requires positive --rows and --dim");
  }
  if (opts.build_algo != "ivf_pq" && opts.build_algo != "nn_descent") {
    usage(argv[0], "--build-algo must be ivf_pq or nn_descent");
  }
  if (opts.n_clusters <= opts.overlap_factor) {
    usage(argv[0], "--clusters must be greater than --overlap");
  }
  if (opts.intermediate_degree <= opts.graph_degree || opts.graph_degree <= 0) {
    usage(argv[0], "intermediate degree must be greater than graph degree, and both positive");
  }
  if (opts.refinement_rate < 1.0f) { usage(argv[0], "--refinement-rate must be at least 1.0"); }
  return opts;
}

size_t checked_product(std::initializer_list<size_t> factors)
{
  size_t result = 1;
  for (auto factor : factors) {
    if (factor != 0 && result > std::numeric_limits<size_t>::max() / factor) {
      throw std::overflow_error("byte-size calculation overflow");
    }
    result *= factor;
  }
  return result;
}

/**
 * A minimal RAII wrapper around a shared mmap.
 *
 * Large all-neighbors outputs cannot be allocated in GPU memory on a 100M run. Mapping them keeps
 * the public cuVS host-output API unchanged while allowing the operating system to page clean data
 * to NVMe. `create` reserves the file up front when the filesystem supports posix_fallocate, so a
 * late SIGBUS caused by a full disk is less likely.
 */
class mapped_file {
 public:
  static mapped_file open_read_only(const std::filesystem::path& path)
  {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::system_error(errno, std::generic_category(), "open " + path.string());
    }
    struct stat stat_buf{};
    if (fstat(fd, &stat_buf) != 0) {
      int error = errno;
      close(fd);
      throw std::system_error(error, std::generic_category(), "fstat " + path.string());
    }
    return mapped_file(path, fd, static_cast<size_t>(stat_buf.st_size), PROT_READ);
  }

  static mapped_file create(const std::filesystem::path& path, size_t bytes)
  {
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      throw std::system_error(errno, std::generic_category(), "open " + path.string());
    }
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      int error = errno;
      close(fd);
      throw std::system_error(error, std::generic_category(), "ftruncate " + path.string());
    }
    int allocate_error = posix_fallocate(fd, 0, static_cast<off_t>(bytes));
    if (allocate_error != 0 && allocate_error != EOPNOTSUPP && allocate_error != ENOSYS &&
        allocate_error != EINVAL) {
      close(fd);
      throw std::system_error(
        allocate_error, std::generic_category(), "posix_fallocate " + path.string());
    }
    if (allocate_error != 0) {
      std::cerr << "warning: filesystem cannot preallocate " << path
                << "; free-space exhaustion may be reported later\n";
    }
    return mapped_file(path, fd, bytes, PROT_READ | PROT_WRITE);
  }

  mapped_file(mapped_file&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, MAP_FAILED)),
      bytes_(std::exchange(other.bytes_, 0)),
      writable_(std::exchange(other.writable_, false))
  {
  }

  mapped_file(const mapped_file&)            = delete;
  mapped_file& operator=(const mapped_file&) = delete;
  mapped_file& operator=(mapped_file&&)      = delete;

  ~mapped_file()
  {
    if (data_ != MAP_FAILED) { munmap(data_, bytes_); }
    if (fd_ >= 0) { close(fd_); }
  }

  template <typename T>
  T* data(size_t byte_offset = 0)
  {
    if (byte_offset > bytes_) { throw std::out_of_range("mmap byte offset"); }
    return reinterpret_cast<T*>(static_cast<std::byte*>(data_) + byte_offset);
  }

  template <typename T>
  const T* data(size_t byte_offset = 0) const
  {
    if (byte_offset > bytes_) { throw std::out_of_range("mmap byte offset"); }
    return reinterpret_cast<const T*>(static_cast<const std::byte*>(data_) + byte_offset);
  }

  [[nodiscard]] size_t size_bytes() const { return bytes_; }

  void flush()
  {
    if (!writable_) { return; }
    if (msync(data_, bytes_, MS_SYNC) != 0) {
      throw std::system_error(errno, std::generic_category(), "msync " + path_.string());
    }
  }

 private:
  mapped_file(std::filesystem::path path, int fd, size_t bytes, int protection)
    : path_(std::move(path)), fd_(fd), bytes_(bytes), writable_((protection & PROT_WRITE) != 0)
  {
    if (bytes_ == 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("cannot mmap an empty file: " + path_.string());
    }
    data_ = mmap(nullptr, bytes_, protection, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      int error = errno;
      close(fd_);
      fd_ = -1;
      throw std::system_error(error, std::generic_category(), "mmap " + path_.string());
    }
  }

  std::filesystem::path path_;
  int fd_        = -1;
  void* data_    = MAP_FAILED;
  size_t bytes_  = 0;
  bool writable_ = false;
};

double seconds_since(clock_type::time_point start)
{
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::string format_gib(size_t bytes)
{
  std::ostringstream output;
  output << std::fixed << std::setprecision(2)
         << static_cast<double>(bytes) / static_cast<double>(1ull << 30) << " GiB";
  return output.str();
}

void resolve_dataset_shape(options& opts, const mapped_file& dataset)
{
  size_t header_bytes = opts.dataset_format == "fbin" ? 2 * sizeof(uint32_t) : 0;
  if (opts.dataset_format == "fbin") {
    if (dataset.size_bytes() < header_bytes) {
      throw std::runtime_error("fbin header is truncated");
    }
    auto header       = dataset.data<uint32_t>();
    int64_t file_rows = header[0];
    int64_t file_dim  = header[1];
    if (opts.rows != 0 && opts.rows != file_rows) {
      throw std::runtime_error("--rows does not match fbin header");
    }
    if (opts.dim != 0 && opts.dim != file_dim) {
      throw std::runtime_error("--dim does not match fbin header");
    }
    opts.rows = file_rows;
    opts.dim  = file_dim;
  }
  size_t expected =
    header_bytes +
    checked_product({static_cast<size_t>(opts.rows), static_cast<size_t>(opts.dim), sizeof(float)});
  if (dataset.size_bytes() != expected) {
    throw std::runtime_error("dataset byte size is " + std::to_string(dataset.size_bytes()) +
                             ", expected " + std::to_string(expected));
  }
  if (static_cast<uint64_t>(opts.rows) > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("CAGRA uint32 graph IDs support at most UINT32_MAX rows");
  }
}

void resolve_gpus(options& opts)
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    throw std::runtime_error("no CUDA GPUs are visible");
  }
  if (opts.gpu_ids.empty()) {
    for (int device = 0; device < count; ++device) {
      opts.gpu_ids.push_back(device);
    }
  }
  for (int device : opts.gpu_ids) {
    if (device < 0 || device >= count) {
      throw std::runtime_error("GPU id " + std::to_string(device) + " is not visible");
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      throw std::runtime_error("cannot query GPU " + std::to_string(device));
    }
    std::cout << "GPU " << device << ": " << prop.name
              << ", memory=" << format_gib(prop.totalGlobalMem) << '\n';
  }
  if (opts.optimize_gpu < 0) { opts.optimize_gpu = opts.gpu_ids.front(); }
  if (std::find(opts.gpu_ids.begin(), opts.gpu_ids.end(), opts.optimize_gpu) ==
      opts.gpu_ids.end()) {
    throw std::runtime_error("--optimize-gpu must also appear in --gpu-ids");
  }
  if (opts.n_clusters < opts.gpu_ids.size()) {
    throw std::runtime_error("--clusters must be at least the number of GPUs");
  }
}

cuvs::neighbors::all_neighbors::all_neighbors_params make_all_neighbors_params(const options& opts,
                                                                               int64_t raw_degree)
{
  namespace alln = cuvs::neighbors::all_neighbors;
  alln::all_neighbors_params params;
  params.n_clusters     = opts.n_clusters;
  params.overlap_factor = opts.overlap_factor;
  params.metric         = cuvs::distance::DistanceType::L2Expanded;

  if (opts.build_algo == "ivf_pq") {
    auto extents = raft::make_extents<int64_t>(opts.rows, opts.dim);
    alln::graph_build_params::ivf_pq_params ivf_pq(extents);
    ivf_pq.refinement_rate = opts.refinement_rate;
    if (opts.ivf_pq_search_batch != 0) {
      ivf_pq.search_params.max_internal_batch_size = opts.ivf_pq_search_batch;
    }
    if (opts.ivf_pq_n_lists != 0) { ivf_pq.build_params.n_lists = opts.ivf_pq_n_lists; }
    params.graph_build_params = ivf_pq;
  } else {
    alln::graph_build_params::nn_descent_params nn_descent(raw_degree);
    nn_descent.max_iterations = opts.nn_descent_iterations;
    params.graph_build_params = nn_descent;
  }
  return params;
}

struct conversion_stats {
  uint64_t self_edges      = 0;
  uint64_t invalid_edges   = 0;
  uint64_t duplicate_edges = 0;
  uint64_t incomplete_rows = 0;
};

conversion_stats convert_knn_graph(
  const int64_t* input, uint32_t* output, int64_t rows, int64_t input_degree, int64_t output_degree)
{
  std::atomic<uint64_t> self_edges{0};
  std::atomic<uint64_t> invalid_edges{0};
  std::atomic<uint64_t> duplicate_edges{0};
  std::atomic<uint64_t> incomplete_rows{0};

#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    int64_t written = 0;
    for (int64_t col = 0; col < input_degree && written < output_degree; ++col) {
      int64_t candidate = input[row * input_degree + col];
      if (candidate == row) {
        self_edges.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (candidate < 0 || candidate >= rows) {
        invalid_edges.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      bool duplicate = false;
      for (int64_t previous = 0; previous < written; ++previous) {
        if (output[row * output_degree + previous] == static_cast<uint32_t>(candidate)) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        duplicate_edges.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      output[row * output_degree + written++] = static_cast<uint32_t>(candidate);
    }
    if (written != output_degree) {
      incomplete_rows.fetch_add(1, std::memory_order_relaxed);
      while (written < output_degree) {
        output[row * output_degree + written++] = std::numeric_limits<uint32_t>::max();
      }
    }
  }

  return {self_edges.load(), invalid_edges.load(), duplicate_edges.load(), incomplete_rows.load()};
}

void write_manifest(const options& opts,
                    const std::filesystem::path& final_graph,
                    int64_t raw_degree,
                    double all_neighbors_seconds,
                    double convert_seconds,
                    double optimize_seconds,
                    double total_seconds,
                    const conversion_stats& stats)
{
  std::ofstream output(opts.output_dir / "run-manifest.txt");
  if (!output) { throw std::runtime_error("cannot write run-manifest.txt"); }
  output << "pipeline=multi_gpu_all_neighbors_cagra\n"
         << "dataset=" << opts.dataset_path << '\n'
         << "dataset_format=" << opts.dataset_format << '\n'
         << "rows=" << opts.rows << '\n'
         << "dim=" << opts.dim << '\n'
         << "dtype=float32\n"
         << "gpu_ids=";
  for (size_t i = 0; i < opts.gpu_ids.size(); ++i) {
    if (i != 0) output << ',';
    output << opts.gpu_ids[i];
  }
  output << '\n'
         << "optimize_gpu=" << opts.optimize_gpu << '\n'
         << "build_algo=" << opts.build_algo << '\n'
         << "clusters=" << opts.n_clusters << '\n'
         << "overlap_factor=" << opts.overlap_factor << '\n'
         << "raw_degree=" << raw_degree << '\n'
         << "intermediate_degree=" << opts.intermediate_degree << '\n'
         << "graph_degree=" << opts.graph_degree << '\n'
         << "refinement_rate=" << opts.refinement_rate << '\n'
         << "ivf_pq_search_batch=" << opts.ivf_pq_search_batch << '\n'
         << "ivf_pq_n_lists=" << opts.ivf_pq_n_lists << '\n'
         << "nn_descent_iterations=" << opts.nn_descent_iterations << '\n'
         << "all_neighbors_seconds=" << all_neighbors_seconds << '\n'
         << "conversion_seconds=" << convert_seconds << '\n'
         << "optimize_seconds=" << optimize_seconds << '\n'
         << "total_seconds=" << total_seconds << '\n'
         << "removed_self_edges=" << stats.self_edges << '\n'
         << "removed_invalid_edges=" << stats.invalid_edges << '\n'
         << "removed_duplicate_edges=" << stats.duplicate_edges << '\n'
         << "incomplete_rows=" << stats.incomplete_rows << '\n'
         << "final_graph=" << final_graph << '\n'
         << "final_graph_layout=row_major_uint32_global_ids\n"
         << "unified_graph=true\n"
         << "search_ready=false\n"
         << "connectivity_guaranteed=false\n";
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto opts = parse_options(argc, argv);
    std::filesystem::create_directories(opts.output_dir);

    auto dataset_mapping = mapped_file::open_read_only(opts.dataset_path);
    resolve_dataset_shape(opts, dataset_mapping);
    resolve_gpus(opts);

    const size_t dataset_header = opts.dataset_format == "fbin" ? 2 * sizeof(uint32_t) : 0;
    const int64_t raw_degree    = opts.intermediate_degree + 1;
    const size_t raw_bytes      = checked_product(
      {static_cast<size_t>(opts.rows), static_cast<size_t>(raw_degree), sizeof(int64_t)});
    const size_t distance_bytes = checked_product(
      {static_cast<size_t>(opts.rows), static_cast<size_t>(raw_degree), sizeof(float)});
    const size_t intermediate_bytes =
      checked_product({static_cast<size_t>(opts.rows),
                       static_cast<size_t>(opts.intermediate_degree),
                       sizeof(uint32_t)});
    const size_t final_bytes = checked_product(
      {static_cast<size_t>(opts.rows), static_cast<size_t>(opts.graph_degree), sizeof(uint32_t)});

    const auto raw_path =
      opts.output_dir / ("all_neighbors_k" + std::to_string(raw_degree) + ".i64");
    const auto distance_path =
      opts.output_dir / ("all_neighbors_k" + std::to_string(raw_degree) + ".f32");
    const auto intermediate_path =
      opts.output_dir / ("knn_k" + std::to_string(opts.intermediate_degree) + ".u32");
    const auto final_path =
      opts.output_dir / ("cagra_k" + std::to_string(opts.graph_degree) + ".u32");

    std::cout << "Dataset: rows=" << opts.rows << " dim=" << opts.dim << " dtype=float32 bytes="
              << format_gib(checked_product(
                   {static_cast<size_t>(opts.rows), static_cast<size_t>(opts.dim), sizeof(float)}))
              << '\n'
              << "Pipeline: overlapping clusters -> local " << opts.build_algo
              << " -> unified host kNN -> CAGRA prune\n"
              << "Graph degrees: raw=" << raw_degree << " intermediate=" << opts.intermediate_degree
              << " final=" << opts.graph_degree << '\n'
              << "File-backed graph bytes: raw=" << format_gib(raw_bytes)
              << " distances=" << format_gib(distance_bytes)
              << " intermediate=" << format_gib(intermediate_bytes)
              << " final=" << format_gib(final_bytes) << '\n';

    auto total_start             = clock_type::now();
    double all_neighbors_seconds = 0.0;
    {
      auto raw_graph = mapped_file::create(raw_path, raw_bytes);
      auto distances = mapped_file::create(distance_path, distance_bytes);

      auto dataset = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
        dataset_mapping.data<const float>(dataset_header), opts.rows, opts.dim);
      auto raw_graph_view = raft::make_host_matrix_view<int64_t, int64_t, raft::row_major>(
        raw_graph.data<int64_t>(), opts.rows, raw_degree);
      auto distances_view = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
        distances.data<float>(), opts.rows, raw_degree);
      auto params = make_all_neighbors_params(opts, raw_degree);

      std::cout << "All-neighbors build starting: GPUs=" << opts.gpu_ids.size()
                << " clusters=" << opts.n_clusters << " overlap=" << opts.overlap_factor << '\n';
      auto phase_start = clock_type::now();
      {
        // Keep SNMG resources in a nested scope. Releasing every rank's CUDA allocations before
        // the single-GPU CAGRA optimizer avoids stale pools/context allocations reducing headroom.
        raft::device_resources_snmg clique(opts.gpu_ids);
        cuvs::neighbors::all_neighbors::build(
          clique, params, dataset, raw_graph_view, std::make_optional(distances_view));
      }
      all_neighbors_seconds = seconds_since(phase_start);
      std::cout << "All-neighbors build completed: seconds=" << all_neighbors_seconds << '\n';
      raw_graph.flush();
    }
    if (!opts.keep_intermediates) { std::filesystem::remove(distance_path); }

    conversion_stats stats;
    double convert_seconds = 0.0;
    {
      auto raw_graph          = mapped_file::open_read_only(raw_path);
      auto intermediate_graph = mapped_file::create(intermediate_path, intermediate_bytes);
      std::cout << "Graph conversion starting: int64 k=" << raw_degree
                << " -> uint32 k=" << opts.intermediate_degree
                << " (remove self/invalid/duplicate edges)\n";
      auto phase_start = clock_type::now();
      stats            = convert_knn_graph(raw_graph.data<const int64_t>(),
                                intermediate_graph.data<uint32_t>(),
                                opts.rows,
                                raw_degree,
                                opts.intermediate_degree);
      convert_seconds  = seconds_since(phase_start);
      intermediate_graph.flush();
      std::cout << "Graph conversion completed: seconds=" << convert_seconds
                << " self=" << stats.self_edges << " invalid=" << stats.invalid_edges
                << " duplicate=" << stats.duplicate_edges
                << " incomplete_rows=" << stats.incomplete_rows << '\n';
    }
    if (stats.incomplete_rows != 0) {
      throw std::runtime_error(
        "conversion produced incomplete rows; retain intermediates and "
        "inspect the raw graph");
    }
    if (!opts.keep_intermediates) { std::filesystem::remove(raw_path); }

    double optimize_seconds = 0.0;
    {
      auto intermediate_graph = mapped_file::open_read_only(intermediate_path);
      auto final_graph        = mapped_file::create(final_path, final_bytes);
      auto input_view         = raft::make_host_matrix_view<uint32_t, int64_t, raft::row_major>(
        intermediate_graph.data<uint32_t>(), opts.rows, opts.intermediate_degree);
      auto output_view = raft::make_host_matrix_view<uint32_t, int64_t, raft::row_major>(
        final_graph.data<uint32_t>(), opts.rows, opts.graph_degree);

      std::cout << "CAGRA optimize starting: GPU=" << opts.optimize_gpu << ' '
                << opts.intermediate_degree << "->" << opts.graph_degree << '\n';
      if (cudaSetDevice(opts.optimize_gpu) != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed for optimize GPU");
      }
      auto phase_start = clock_type::now();
      raft::resources optimize_resources;
      cuvs::neighbors::cagra::helpers::optimize(optimize_resources, input_view, output_view);
      raft::resource::sync_stream(optimize_resources);
      optimize_seconds = seconds_since(phase_start);
      final_graph.flush();
      std::cout << "CAGRA optimize completed: seconds=" << optimize_seconds << '\n';
    }
    if (!opts.keep_intermediates) { std::filesystem::remove(intermediate_path); }

    double total_seconds = seconds_since(total_start);
    write_manifest(opts,
                   final_path,
                   raw_degree,
                   all_neighbors_seconds,
                   convert_seconds,
                   optimize_seconds,
                   total_seconds,
                   stats);
    std::cout
      << "Unified graph completed: seconds=" << total_seconds << " output=" << final_path
      << " bytes=" << final_bytes << '\n'
      << "This is a graph-only artifact; attach a compatible dataset before CAGRA search.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
