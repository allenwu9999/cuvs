cmake_minimum_required(VERSION 3.26)
project(all_neighbors_refinement_bench LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
set(CUVS_PREFIX /opt/conda CACHE PATH "Installed cuVS dependencies")
set(CUVS_SOURCE /work CACHE PATH "Modified cuVS source")
set(CUVS_BASELINE /baseline CACHE PATH "Original cuVS source")
# This image predates RMM's value()->get() accessor rename. Add an inline
# alias in a private header overlay, identically for CPU and GPU builds.
file(READ "${CUVS_PREFIX}/include/rmm/cuda_stream_view.hpp" stream_header)
if(NOT stream_header MATCHES "cudaStream_t get\\(")
  string(REPLACE "cudaStream_t value() const noexcept;"
    "cudaStream_t value() const noexcept;\n  [[nodiscard]] cudaStream_t get() const noexcept { return value(); }"
    stream_header "${stream_header}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compat/rmm/cuda_stream_view.hpp" "${stream_header}")
endif()
foreach(variant cpu gpu)
  if(variant STREQUAL cpu)
    set(source ${CUVS_BASELINE})
  else()
    set(source ${CUVS_SOURCE})
  endif()
  add_executable(refine_${variant} main.cu
    ${source}/cpp/src/neighbors/all_neighbors/all_neighbors.cu
    ${source}/cpp/src/core/omp_wrapper.cpp)
  target_include_directories(refine_${variant} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/compat ${source}/cpp/include
    ${source}/cpp/src ${CUVS_PREFIX}/include ${CUVS_PREFIX}/include/rapids)
  target_compile_options(refine_${variant} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-fopenmp>)
  target_compile_definitions(refine_${variant} PRIVATE RAFT_SYSTEM_LITTLE_ENDIAN=1)
  set_target_properties(refine_${variant} PROPERTIES BUILD_RPATH "${CUVS_PREFIX}/lib" LINKER_LANGUAGE CXX)
  target_link_libraries(refine_${variant} PRIVATE
    ${CUVS_PREFIX}/lib/libcuvs.so ${CUVS_PREFIX}/lib/librmm.so
    ${CUVS_PREFIX}/lib/librapids_logger.so CUDA::cudart OpenMP::OpenMP_CXX Threads::Threads)
endforeach()

# Focused regression checks for odd dimensions, batch tails, invalid candidates,
# and graph degrees above the general refine API's warp-sort capacity.
add_executable(refine_test EXCLUDE_FROM_ALL refine_test.cu)
target_include_directories(refine_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/compat
  ${CUVS_SOURCE}/cpp/include ${CUVS_PREFIX}/include ${CUVS_PREFIX}/include/rapids)
target_compile_options(refine_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-fopenmp>)
target_compile_definitions(refine_test PRIVATE RAFT_SYSTEM_LITTLE_ENDIAN=1)
set_target_properties(refine_test PROPERTIES BUILD_RPATH "${CUVS_PREFIX}/lib" LINKER_LANGUAGE CXX)
target_link_libraries(refine_test PRIVATE ${CUVS_PREFIX}/lib/libcuvs.so
  ${CUVS_PREFIX}/lib/librmm.so ${CUVS_PREFIX}/lib/librapids_logger.so
  CUDA::cudart OpenMP::OpenMP_CXX Threads::Threads)
