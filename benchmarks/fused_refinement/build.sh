#!/bin/bash
set -euo pipefail
cd /bench
nvcc -O3 -lineinfo -std=c++20 -arch=sm_120 --expt-extended-lambda --expt-relaxed-constexpr \
  -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -Xcompiler=-fopenmp \
  -I/opt/conda/include -I/opt/conda/include/rapids bench.cu \
  -L/opt/conda/lib -lcuvs -lrmm -lrapids_logger \
  -Xlinker=-rpath -Xlinker=/opt/conda/lib -o bench
