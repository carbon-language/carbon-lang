// RUN: %clang -v --sysroot=/tmp/no-cuda-there 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --sysroot=%S/Inputs/CUDA 2>&1 | FileCheck %s
// RUN: %clang -v --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// NOCUDA-NOT: Found CUDA installation:
