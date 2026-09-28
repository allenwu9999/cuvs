#include "baseline_refine.cuh"
#include "fused_refine.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_OK(expr) do { auto error = (expr); if (error != cudaSuccess) \
  throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(error)); } while (0)

__host__ __device__ uint32_t hash32(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; v *= 0x846ca68b; v ^= v >> 16; return v;
}
__host__ __device__ float coordinate(uint64_t i) {
  return float(hash32(uint32_t(i) ^ uint32_t(i >> 32)) & 0xffffff) / 8388608.0f - 1.0f;
}
__global__ void init_data(float* x, size_t size) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
       i += size_t(blockDim.x) * gridDim.x) x[i] = coordinate(i);
}
__global__ void init_candidates(int64_t* candidates, int n, int b, int c, bool invalid) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(b) * c;
       i += size_t(blockDim.x) * gridDim.x) {
    int row = i / c, j = i % c;
    // Odd increment on power-of-two N gives unique, scattered candidate IDs.
    candidates[i] = (uint64_t(hash32(row + 913)) + uint64_t(j) * 104729) % n;
    if (invalid && j == c - 1) candidates[i] = -1;
    if (invalid && j == c - 2) candidates[i] = n;
  }
}

__global__ void compare_outputs(const float* ref_d, const int64_t* ref_i,
                                const float* d, const int64_t* ids,
                                int n, int b, int k, unsigned long long* counters) {
  for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x; p < size_t(b) * k;
       p += size_t(blockDim.x) * gridDim.x) {
    if (!isfinite(d[p]) || ids[p] < 0 || ids[p] >= n ||
        fabsf(ref_d[p] - d[p]) > 2e-5f * fmaxf(1.0f, fabsf(ref_d[p])) ||
        (p % k && d[p] < d[p - 1])) atomicAdd(counters, 1ULL);
    if (ids[p] != ref_i[p]) atomicAdd(counters + 1, 1ULL);
  }
}

// Identical distance kernel, batch size, and select_k call to the commit.
// Allow a query prefix B <= N to also measure ordinary refinement-sized batches.
void baseline(const raft::resources& res, const float* data, const int64_t* cand,
              float* scratch, int64_t* ids, float* dist, int64_t n, int b, int d, int c, int k) {
  if (b == n) {
    cuvs::neighbors::all_neighbors::detail::refine_candidates(res,
      raft::make_device_matrix_view<const float, int64_t>(data, n, d),
      raft::make_device_matrix_view<const int64_t, int64_t>(cand, b, c),
      raft::make_device_matrix_view<float, int64_t>(scratch, b, c),
      raft::make_device_matrix_view<int64_t, int64_t>(ids, b, k),
      raft::make_device_matrix_view<float, int64_t>(dist, b, k));
    return;
  }
  auto stream = raft::resource::get_cuda_stream(res);
  for (int offset = 0; offset < b; offset += 8192) {
    int rows = std::min(8192, b - offset);
    size_t pairs = size_t(rows) * c;
    cuvs::neighbors::all_neighbors::detail::candidate_l2_distances<float, int64_t>
      <<<(pairs + 7) / 8, 256, 0, stream>>>(data, cand + size_t(offset) * c,
        scratch + size_t(offset) * c, n, d, offset, pairs, c);
    cuvs::selection::select_k(res,
      raft::make_device_matrix_view<const float, int64_t>(scratch + size_t(offset) * c, rows, c),
      std::make_optional(raft::make_device_matrix_view<const int64_t, int64_t>(cand + size_t(offset) * c, rows, c)),
      raft::make_device_matrix_view<float, int64_t>(dist + size_t(offset) * k, rows, k),
      raft::make_device_matrix_view<int64_t, int64_t>(ids + size_t(offset) * k, rows, k), true, true);
  }
  CUDA_OK(cudaPeekAtLastError());
}

int main(int argc, char** argv) try {
  int n = 1048576, b = 65536, d = 128, c = 128, k = 64, rounds = 9, repeats = 20;
  bool invalid = false;
  std::string file;
  std::vector<int> methods{0, 4, 8, 16, 32};
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--invalid") invalid = true;
    else if (i + 1 >= argc) throw std::runtime_error("missing argument");
    else if (arg == "--rows") n = std::stoi(argv[++i]);
    else if (arg == "--queries") b = std::stoi(argv[++i]);
    else if (arg == "--dim") d = std::stoi(argv[++i]);
    else if (arg == "--candidates") c = std::stoi(argv[++i]);
    else if (arg == "--k") k = std::stoi(argv[++i]);
    else if (arg == "--rounds") rounds = std::stoi(argv[++i]);
    else if (arg == "--repeats") repeats = std::stoi(argv[++i]);
    else if (arg == "--data") file = argv[++i];
    else if (arg == "--methods") {
      methods.clear(); std::stringstream ss(argv[++i]); std::string item;
      while(std::getline(ss,item,',')) methods.push_back(std::stoi(item));
    }
    else throw std::runtime_error("unknown argument " + arg);
  }
  if (b > n || k > 256 || c < k + (invalid ? 2 : 0) || k < 1 || b < 1 || c > n)
    throw std::runtime_error("unsupported shape");
  if (std::find(methods.begin(), methods.end(), 0) == methods.end()) throw std::runtime_error("baseline method 0 required");
  for(int m:methods) if(m!=0 && m!=4 && m!=8 && m!=16 && m!=32 && m!=108 && m!=116 && m!=132)
    throw std::runtime_error("unknown method");
  std::vector<float> host_data;
  if (!file.empty()) {
    std::ifstream input(file, std::ios::binary);
    int32_t header[2]; input.read(reinterpret_cast<char*>(header), 8);
    if (!input || header[0] < n) throw std::runtime_error("bad fbin or too few rows");
    d = header[1];
    host_data.resize(size_t(n) * d);
    input.read(reinterpret_cast<char*>(host_data.data()), host_data.size() * sizeof(float));
    if (!input) throw std::runtime_error("short fbin read");
  }
  rmm::mr::pool_memory_resource pool(rmm::mr::get_current_device_resource_ref(), size_t(2) << 30);
  struct ResourceRestore {
    cuda::mr::any_resource<cuda::mr::device_accessible> old;
    ~ResourceRestore() { rmm::mr::set_current_device_resource(old); }
  } restore{rmm::mr::set_current_device_resource({pool})};
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto x = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto candidates = raft::make_device_matrix<int64_t, int64_t>(res, b, c);
  auto scratch = raft::make_device_matrix<float, int64_t>(res, b, c);
  auto ids = raft::make_device_matrix<int64_t, int64_t>(res, b, k);
  auto dist = raft::make_device_matrix<float, int64_t>(res, b, k);
  auto ref_ids = raft::make_device_matrix<int64_t, int64_t>(res, b, k);
  auto ref_dist = raft::make_device_matrix<float, int64_t>(res, b, k);
  auto counters = raft::make_device_vector<unsigned long long, int64_t>(res, 2);
  if (host_data.empty()) {
    init_data<<<4096, 256, 0, stream>>>(x.data_handle(), size_t(n) * d);
    init_candidates<<<4096, 256, 0, stream>>>(candidates.data_handle(), n, b, c, invalid);
  } else {
    raft::update_device(x.data_handle(), host_data.data(), host_data.size(), stream);
    cuvs::neighbors::ivf_pq::index_params ip;
    ip.n_lists = std::max(16, n / 2000); ip.pq_dim = 64; ip.pq_bits = 4;
    ip.kmeans_n_iters = 10; ip.kmeans_trainset_fraction = std::min(1.0, 100000.0 / n);
    ip.metric = cuvs::distance::DistanceType::L2Expanded;
    std::cerr << "Preparing IVF-PQ candidates outside timing\n";
    auto index = cuvs::neighbors::ivf_pq::build(res, ip, raft::make_const_mdspan(x.view()));
    cuvs::neighbors::ivf_pq::search_params sp;
    sp.n_probes = 16; sp.max_internal_batch_size = 8192;
    sp.lut_dtype = CUDA_R_32F; sp.internal_distance_dtype = CUDA_R_32F;
    sp.coarse_search_dtype = CUDA_R_32F;
    cuvs::neighbors::ivf_pq::search(res, sp, index,
      raft::make_device_matrix_view<const float, int64_t>(x.data_handle(), b, d),
      candidates.view(), scratch.view());
    raft::resource::sync_stream(res);
  }
  CUDA_OK(cudaPeekAtLastError());
  auto run = [&](int method, bool reference = false) {
    auto oi = reference ? ref_ids.data_handle() : ids.data_handle();
    auto od = reference ? ref_dist.data_handle() : dist.data_handle();
    if (method == 0) baseline(res, x.data_handle(), candidates.data_handle(), scratch.data_handle(), oi, od, n, b, d, c, k);
    else if (method == 4) fusion_experiment::dispatch<4>(x.data_handle(), candidates.data_handle(), oi, od, n, b, d, c, k, stream);
    else if (method == 8) fusion_experiment::dispatch<8>(x.data_handle(), candidates.data_handle(), oi, od, n, b, d, c, k, stream);
    else if (method == 16) fusion_experiment::dispatch<16>(x.data_handle(), candidates.data_handle(), oi, od, n, b, d, c, k, stream);
    else if (method == 32) fusion_experiment::dispatch<32>(x.data_handle(), candidates.data_handle(), oi, od, n, b, d, c, k, stream);
    else for(int offset=0;offset<b;offset+=8192) {
      int rows=std::min(8192,b-offset);
      if(method==108) fusion_experiment::dispatch<8>(x.data_handle(), candidates.data_handle(), oi, od, n, rows, d, c, k, stream, offset);
      else if(method==116) fusion_experiment::dispatch<16>(x.data_handle(), candidates.data_handle(), oi, od, n, rows, d, c, k, stream, offset);
      else fusion_experiment::dispatch<32>(x.data_handle(), candidates.data_handle(), oi, od, n, rows, d, c, k, stream, offset);
    }
    CUDA_OK(cudaPeekAtLastError());
  };
  run(0, true);
  raft::resource::sync_stream(res);
  std::vector<unsigned long long> id_mismatch(133);
  // CPU FP64 candidate reference for deterministic queries spread across B.
  auto validate_cpu = [&]() {
    const int ns = std::min(32, b);
    for (int s = 0; s < ns; ++s) {
      int row = ns == 1 ? 0 : int(int64_t(s) * (b - 1) / (ns - 1));
      std::vector<int64_t> ci(c), oi(k); std::vector<float> od(k);
      CUDA_OK(cudaMemcpy(ci.data(), candidates.data_handle() + size_t(row)*c, c*8, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(oi.data(), ids.data_handle() + size_t(row)*k, k*8, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(od.data(), dist.data_handle() + size_t(row)*k, k*4, cudaMemcpyDeviceToHost));
      auto value = [&](int64_t id, int col) { return host_data.empty() ? coordinate(uint64_t(id)*d+col) : host_data[size_t(id)*d+col]; };
      auto exact = [&](int64_t id) {
        if (id < 0 || id >= n) return INFINITY + double(0);
        double sum=0; for(int j=0;j<d;++j) {double df=double(value(row,j))-double(value(id,j));sum+=df*df;} return sum;
      };
      std::vector<double> expected; for(auto id:ci) expected.push_back(exact(id));
      std::sort(expected.begin(), expected.end());
      for(int j=0;j<k;++j) {
        double tolerance=2e-5*std::max(1.0,expected[j]);
        if(!std::isfinite(od[j]) || std::abs(od[j]-expected[j])>tolerance ||
           std::abs(od[j]-exact(oi[j]))>tolerance || std::find(ci.begin(),ci.end(),oi[j])==ci.end())
          throw std::runtime_error("CPU reference validation failed at row "+std::to_string(row));
      }
      std::sort(oi.begin(),oi.end());
      if(std::adjacent_find(oi.begin(),oi.end())!=oi.end()) throw std::runtime_error("duplicate output ID");
    }
  };
  for (int method: methods) {
    run(method);
    CUDA_OK(cudaMemsetAsync(counters.data_handle(), 0, 16, stream));
    compare_outputs<<<4096, 256, 0, stream>>>(ref_dist.data_handle(), ref_ids.data_handle(), dist.data_handle(), ids.data_handle(), n, b, k, counters.data_handle());
    unsigned long long counts[2];
    raft::update_host(counts, counters.data_handle(), 2, stream);
    raft::resource::sync_stream(res);
    if(counts[0]) throw std::runtime_error("Full-output comparison failed: method="+std::to_string(method)+" errors="+std::to_string(counts[0]));
    id_mismatch[method]=counts[1]; validate_cpu();
    for(int w=0;w<3;++w) run(method);
  }
  raft::resource::sync_stream(res);
  std::cerr << "Correctness passed: n="<<n<<" b="<<b<<" d="<<d<<" c="<<c<<" k="<<k<<"\n";
  cudaEvent_t start, stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
  std::vector<std::vector<double>> gpu_times(133), wall_times(133);
  std::mt19937 rng(1729);
  for(int round=0;round<rounds;++round) {
    std::shuffle(methods.begin(),methods.end(),rng);
    for(int method:methods) {
      raft::resource::sync_stream(res);
      auto wall_start=std::chrono::steady_clock::now();
      CUDA_OK(cudaEventRecord(start,stream));
      for(int rep=0;rep<repeats;++rep) run(method);
      CUDA_OK(cudaEventRecord(stop,stream)); CUDA_OK(cudaEventSynchronize(stop));
      float ms; CUDA_OK(cudaEventElapsedTime(&ms,start,stop));
      gpu_times[method].push_back(ms/repeats);
      wall_times[method].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count()/repeats);
      std::cerr<<"SAMPLE,"<<round<<","<<method<<","<<std::setprecision(9)<<gpu_times[method].back()<<","<<wall_times[method].back()<<"\n";
    }
  }
  CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
  auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
  std::cout<<std::setprecision(9);
  std::cout<<"source,n,b,d,c,k,method,gpu_median_ms,gpu_min_ms,gpu_max_ms,wall_median_ms,speedup,id_mismatches,rounds,repeats\n";
  std::sort(methods.begin(),methods.end());
  for(int method:methods) {
    auto& v=gpu_times[method];
    auto label=method==0?"baseline":"fused_t"+std::to_string(method%100)+(method>=100?"_batch8192":"");
    std::cout<<(file.empty()?"synthetic":"ivfpq")<<","<<n<<","<<b<<","<<d<<","<<c<<","<<k<<","<<label<<","<<median(v)<<","<<*std::min_element(v.begin(),v.end())<<","<<*std::max_element(v.begin(),v.end())<<","<<median(wall_times[method])<<","<<median(gpu_times[0])/median(v)<<","<<id_mismatch[method]<<","<<rounds<<","<<repeats<<"\n";
  }
  return 0;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
