// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 vector compare absolute intrinsics

#include <arm_neon.h>

uint32x2_t test_vcale_f32(float32x2_t a1, float32x2_t a2) {
  // CHECK: test_vcale_f32
  return vcale_f32(a1, a2);
  // CHECK: llvm.arm64.neon.facge.v2i32.v2f32
  // no check for ret here, as there is a bitcast
}

uint32x4_t test_vcaleq_f32(float32x4_t a1, float32x4_t a2) {
  // CHECK: test_vcaleq_f32
  return vcaleq_f32(a1, a2);
  // CHECK: llvm.arm64.neon.facge.v4i32.v4f32{{.*a2,.*a1}}
  // no check for ret here, as there is a bitcast
}

uint32x2_t test_vcalt_f32(float32x2_t a1, float32x2_t a2) {
  // CHECK: test_vcalt_f32
  return vcalt_f32(a1, a2);
  // CHECK: llvm.arm64.neon.facgt.v2i32.v2f32{{.*a2,.*a1}}
  // no check for ret here, as there is a bitcast
}

uint32x4_t test_vcaltq_f32(float32x4_t a1, float32x4_t a2) {
  // CHECK: test_vcaltq_f32
  return vcaltq_f32(a1, a2);
  // CHECK: llvm.arm64.neon.facgt.v4i32.v4f32{{.*a2,.*a1}}
}

uint64x2_t test_vcagtq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vcagtq_f64
  return vcagtq_f64(a1, a2);
  // CHECK: llvm.arm64.neon.facgt.v2i64.v2f64{{.*a1,.*a2}}
  // no check for ret here, as there is a bitcast
}

uint64x2_t test_vcaltq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vcaltq_f64
  return vcaltq_f64(a1, a2);
  // CHECK: llvm.arm64.neon.facgt.v2i64.v2f64{{.*a2,.*a1}}
  // no check for ret here, as there is a bitcast
}

uint64x2_t test_vcageq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vcageq_f64
  return vcageq_f64(a1, a2);
  // CHECK: llvm.arm64.neon.facge.v2i64.v2f64{{.*a1,.*a2}}
  // no check for ret here, as there is a bitcast
}

uint64x2_t test_vcaleq_f64(float64x2_t a1, float64x2_t a2) {
  // CHECK: test_vcaleq_f64
  return vcaleq_f64(a1, a2);
  // CHECK: llvm.arm64.neon.facge.v2i64.v2f64{{.*a2,.*a1}}
  // no check for ret here, as there is a bitcast
}
