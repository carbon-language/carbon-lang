// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
//
// # Check that we properly detect CUDA installation.
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/no-cuda-there 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/Inputs/CUDA 2>&1 | FileCheck %s
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s

// Verify that CUDA include path gets added
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix CUDAINC
// Verify that -nocudainc disables CUDA include paths.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   -nocudainc --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC
// We should not add any CUDA include paths if there's no valid CUDA installation
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// NOCUDA-NOT: Found CUDA installation:

// COMMON: "-triple" "nvptx-nvidia-cuda"
// COMMON-SAME: "-fcuda-is-device"
// CUDAINC-SAME: "-internal-isystem" "{{.*}}/Inputs/CUDA/usr/local/cuda/include"
// NOCUDAINC-NOT: "-internal-isystem" "{{.*}}/cuda/include"
// COMMON-SAME: "-x" "cuda"

