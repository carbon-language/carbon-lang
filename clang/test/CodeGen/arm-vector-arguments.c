// RUN: %clang_cc1 -triple thumbv7-apple-darwin9 \
// RUN:   -target-abi apcs-gnu \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi soft \
// RUN:   -target-feature +soft-float-abi \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

#include <arm_neon.h>

// CHECK: define void @f0(%struct.__simd128_int8_t* sret %agg.result, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
int8x16_t f0(int8x16_t a0, int8x16_t a1) {
  return vzipq_s8(a0, a1).val[0];
}
