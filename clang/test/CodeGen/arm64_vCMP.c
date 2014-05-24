// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s

// Test ARM64 SIMD fused multiply add intrinsics

#include <arm_neon.h>

int64x2_t test_vabsq_s64(int64x2_t a1) {
  // CHECK: test_vabsq_s64
  return vabsq_s64(a1);
  // CHECK: llvm.aarch64.neon.abs.v2i64
  // CHECK-NEXT: ret
}

int64_t test_vceqd_s64(int64_t a1, int64_t a2) {
  // CHECK: test_vceqd_s64
  return vceqd_s64(a1, a2);
  // CHECK: [[BIT:%[0-9a-zA-Z.]+]] = icmp eq i64 %a1, %a2
  // CHECK: sext i1 [[BIT]] to i64
}

int64_t test_vceqd_f64(float64_t a1, float64_t a2) {
  // CHECK: test_vceqd_f64
  return vceqd_f64(a1, a2);
  // CHECK: [[BIT:%[0-9a-zA-Z.]+]] = fcmp oeq double %a1, %a2
  // CHECK: sext i1 [[BIT]] to i64
}

uint64_t test_vcgtd_u64(uint64_t a1, uint64_t a2) {
  // CHECK: test_vcgtd_u64
  return vcgtd_u64(a1, a2);
  // CHECK: [[BIT:%[0-9a-zA-Z.]+]] = icmp ugt i64 %a1, %a2
  // CHECK: sext i1 [[BIT]] to i64
}

uint64_t test_vcled_u64(uint64_t a1, uint64_t a2) {
  // CHECK: test_vcled_u64
  return vcled_u64(a1, a2);
  // CHECK: [[BIT:%[0-9a-zA-Z.]+]] = icmp ule i64 %a1, %a2
  // CHECK: sext i1 [[BIT]] to i64
}

int64_t test_vceqzd_s64(int64_t a1) {
  // CHECK: test_vceqzd_s64
  return vceqzd_s64(a1);
  // CHECK: [[BIT:%[0-9a-zA-Z.]+]] = icmp eq i64 %a1, 0
  // CHECK: sext i1 [[BIT]] to i64
}

uint64x2_t test_vceqq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vceqq_u64
  return vceqq_u64(a1, a2);
  // CHECK:  icmp eq <2 x i64> %a1, %a2
}

uint64x2_t test_vcgeq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vcgeq_s64
  return vcgeq_s64(a1, a2);
  // CHECK:  icmp sge <2 x i64> %a1, %a2
}

uint64x2_t test_vcgeq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vcgeq_u64
  return vcgeq_u64(a1, a2);
  // CHECK:  icmp uge <2 x i64> %a1, %a2
}

uint64x2_t test_vcgtq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vcgtq_s64
  return vcgtq_s64(a1, a2);
  // CHECK: icmp sgt <2 x i64> %a1, %a2
}

uint64x2_t test_vcgtq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vcgtq_u64
  return vcgtq_u64(a1, a2);
  // CHECK: icmp ugt <2 x i64> %a1, %a2
}

uint64x2_t test_vcleq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vcleq_s64
  return vcleq_s64(a1, a2);
  // CHECK: icmp sle <2 x i64> %a1, %a2
}

uint64x2_t test_vcleq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vcleq_u64
  return vcleq_u64(a1, a2);
  // CHECK: icmp ule <2 x i64> %a1, %a2
}

uint64x2_t test_vcltq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vcltq_s64
  return vcltq_s64(a1, a2);
  // CHECK: icmp slt <2 x i64> %a1, %a2
}

uint64x2_t test_vcltq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vcltq_u64
  return vcltq_u64(a1, a2);
  // CHECK: icmp ult <2 x i64> %a1, %a2
}

int64x2_t test_vqabsq_s64(int64x2_t a1) {
  // CHECK: test_vqabsq_s64
  return vqabsq_s64(a1);
  // CHECK: llvm.aarch64.neon.sqabs.v2i64(<2 x i64> %a1)
  // CHECK-NEXT: ret
}
