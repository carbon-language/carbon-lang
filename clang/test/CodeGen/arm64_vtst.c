// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD comparison test intrinsics

#include <arm_neon.h>

uint64x2_t test_vtstq_s64(int64x2_t a1, int64x2_t a2) {
  // CHECK: test_vtstq_s64
  return vtstq_s64(a1, a2);
  // CHECK: [[COMMONBITS:%[A-Za-z0-9.]+]] = and <2 x i64> %a1, %a2
  // CHECK: [[MASK:%[A-Za-z0-9.]+]] = icmp ne <2 x i64> [[COMMONBITS]], zeroinitializer
  // CHECK: [[RES:%[A-Za-z0-9.]+]] = sext <2 x i1> [[MASK]] to <2 x i64>
  // CHECK: ret <2 x i64> [[RES]]
}

uint64x2_t test_vtstq_u64(uint64x2_t a1, uint64x2_t a2) {
  // CHECK: test_vtstq_u64
  return vtstq_u64(a1, a2);
  // CHECK: [[COMMONBITS:%[A-Za-z0-9.]+]] = and <2 x i64> %a1, %a2
  // CHECK: [[MASK:%[A-Za-z0-9.]+]] = icmp ne <2 x i64> [[COMMONBITS]], zeroinitializer
  // CHECK: [[RES:%[A-Za-z0-9.]+]] = sext <2 x i1> [[MASK]] to <2 x i64>
  // CHECK: ret <2 x i64> [[RES]]
}
