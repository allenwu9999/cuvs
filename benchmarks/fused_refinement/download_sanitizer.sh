#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p tools
cd tools

archive=cuda_sanitizer_api-linux-x86_64-12.9.79-archive.tar.xz
curl --fail --location --output "$archive" \
  "https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/$archive"
echo "e23aad21132ff58b92a22aad372a7048793400b79c625665d325d4ecec6979bf  $archive" | sha256sum --check
tar -xf "$archive"
