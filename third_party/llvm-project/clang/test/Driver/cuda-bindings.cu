// Tests the bindings generated for a CUDA offloading target for different
// combinations of:
// - Number of gpu architectures;
// - Host/device-only compilation;
// - User-requested final phase - binary or assembly.
// It parallels cuda-phases.cu test, but verifies whether output file is temporary or not.

// It's hard to check whether file name is temporary in a portable
// way. Instead we check whether we've generated a permanent name on
// device side, which appends '-device-cuda-<triple>' suffix.

// REQUIRES: clang-driver
// REQUIRES: powerpc-registered-target
// REQUIRES: nvptx-registered-target

//
// Test single gpu architecture with complete compilation.
// No intermediary device files should have "-device-cuda..." in the name.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings --cuda-gpu-arch=sm_30 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN %s
// BIN: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "nvptx64-nvidia-cuda" - "NVPTX::Linker",{{.*}} output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// BIN-NOT: cuda-bindings-device-cuda-nvptx64
// BIN: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"

//
// Test single gpu architecture up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings --cuda-gpu-arch=sm_30 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM %s
// ASM-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// ASM-DAG: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test two gpu architectures with complete compilation.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s 2>&1 \
// RUN: | FileCheck -check-prefix=BIN2 %s
// BIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Linker",{{.*}} output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// BIN2-NOT: cuda-bindings-device-cuda-nvptx64
// BIN2: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"

//
// Test two gpu architectures up to the assemble phase.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s -S 2>&1 \
// RUN: | FileCheck -check-prefix=ASM2 %s
// ASM2-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// ASM2-DAG: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.s"
// ASM2-DAG: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test one or more gpu architecture with complete compilation in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only 2>&1 \
// RUN: | FileCheck -check-prefix=HBIN %s
// HBIN: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}}  output:
// HBIN-NOT: cuda-bindings-device-cuda-nvptx64
// HBIN: # "powerpc64le-ibm-linux-gnu" - "GNU::Linker", inputs:{{.*}}, output: "a.out"

//
// Test one or more gpu architecture up to the assemble phase in host-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-host-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=HASM %s
// HASM: # "powerpc64le-ibm-linux-gnu" - "clang",{{.*}} output: "cuda-bindings.s"

//
// Test single gpu architecture with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN %s
// DBIN: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.o"

//
// Test single gpu architecture up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM %s
// DASM: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"

//
// Test two gpu architectures with complete compilation in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only 2>&1 \
// RUN: | FileCheck -check-prefix=DBIN2 %s
// DBIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN2-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.o"
// DBIN2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output:
// DBIN2-NOT: cuda-bindings-device-cuda-nvptx64
// DBIN2: # "nvptx64-nvidia-cuda" - "NVPTX::Assembler",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.o"

//
// Test two gpu architectures up to the assemble phase in device-only
// compilation mode.
//
// RUN: %clang -target powerpc64le-ibm-linux-gnu -ccc-print-bindings \
// RUN:        --cuda-gpu-arch=sm_30 --cuda-gpu-arch=sm_35 %s --cuda-device-only -S 2>&1 \
// RUN: | FileCheck -check-prefix=DASM2 %s
// DASM2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_30.s"
// DASM2: # "nvptx64-nvidia-cuda" - "clang",{{.*}} output: "cuda-bindings-cuda-nvptx64-nvidia-cuda-sm_35.s"
