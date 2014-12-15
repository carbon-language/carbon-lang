// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s

#include "Inputs/cuda.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_BLOCKS_PER_MP     2

// Test both max threads per block and Min cta per sm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
Kernel1()
{
}
}

// CHECK: !{{[0-9]+}} = !{void ()* @Kernel1, !"maxntidx", i32 256}
// CHECK: !{{[0-9]+}} = !{void ()* @Kernel1, !"minctasm", i32 2}

// Test only max threads per block. Min cta per sm defaults to 0, and
// CodeGen doesn't output a zero value for minctasm.
extern "C" {
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK )
Kernel2()
{
}
}

// CHECK: !{{[0-9]+}} = !{void ()* @Kernel2, !"maxntidx", i32 256}
