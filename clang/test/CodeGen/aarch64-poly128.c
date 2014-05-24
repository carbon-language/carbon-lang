// REQUIRES: arm64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -ffp-contract=fast -S -O3 -o - %s | FileCheck %s --check-prefix=CHECK \
// RUN:  --check-prefix=CHECK-ARM64

// Test new aarch64 intrinsics with poly128
// FIXME: Currently, poly128_t equals to uint128, which will be spilt into
// two 64-bit GPR(eg X0, X1). Now moving data from X0, X1 to FPR128 will
// introduce 2 store and 1 load instructions(store X0, X1 to memory and
// then load back to Q0). If target has NEON, this is better replaced by
// FMOV or INS.

#include <arm_neon.h>

void test_vstrq_p128(poly128_t * ptr, poly128_t val) {
  // CHECK-LABEL: test_vstrq_p128
  vstrq_p128(ptr, val);

  // CHECK-ARM64: stp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
}

poly128_t test_vldrq_p128(poly128_t * ptr) {
  // CHECK-LABEL: test_vldrq_p128
  return vldrq_p128(ptr);

  // CHECK-ARM64: ldp {{x[0-9]+}}, {{x[0-9]+}}, [x0]
}

void test_ld_st_p128(poly128_t * ptr) {
  // CHECK-LABEL: test_ld_st_p128
   vstrq_p128(ptr+1, vldrq_p128(ptr));

 // CHECK-ARM64: ldp [[PLO:x[0-9]+]], [[PHI:x[0-9]+]], [{{x[0-9]+}}]
 // CHECK-ARM64-NEXT: stp [[PLO]], [[PHI]], [{{x[0-9]+}}, #16]
}

poly128_t test_vmull_p64(poly64_t a, poly64_t b) {
  // CHECK-LABEL: test_vmull_p64
  return vmull_p64(a, b);
  // CHECK: pmull {{v[0-9]+}}.1q, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d
}

poly128_t test_vmull_high_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vmull_high_p64
  return vmull_high_p64(a, b);
  // CHECK: pmull2 {{v[0-9]+}}.1q, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

// CHECK-LABEL: test_vreinterpretq_p128_s8
// CHECK: ret
poly128_t test_vreinterpretq_p128_s8(int8x16_t a) {
  return vreinterpretq_p128_s8(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_s16
// CHECK: ret
poly128_t test_vreinterpretq_p128_s16(int16x8_t a) {
  return vreinterpretq_p128_s16(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_s32
// CHECK: ret
poly128_t test_vreinterpretq_p128_s32(int32x4_t a) {
  return vreinterpretq_p128_s32(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_s64
// CHECK: ret
poly128_t test_vreinterpretq_p128_s64(int64x2_t a) {
  return vreinterpretq_p128_s64(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_u8
// CHECK: ret
poly128_t test_vreinterpretq_p128_u8(uint8x16_t a) {
  return vreinterpretq_p128_u8(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_u16
// CHECK: ret
poly128_t test_vreinterpretq_p128_u16(uint16x8_t a) {
  return vreinterpretq_p128_u16(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_u32
// CHECK: ret
poly128_t test_vreinterpretq_p128_u32(uint32x4_t a) {
  return vreinterpretq_p128_u32(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_u64
// CHECK: ret
poly128_t test_vreinterpretq_p128_u64(uint64x2_t a) {
  return vreinterpretq_p128_u64(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_f32
// CHECK: ret
poly128_t test_vreinterpretq_p128_f32(float32x4_t a) {
  return vreinterpretq_p128_f32(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_f64
// CHECK: ret
poly128_t test_vreinterpretq_p128_f64(float64x2_t a) {
  return vreinterpretq_p128_f64(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_p8
// CHECK: ret
poly128_t test_vreinterpretq_p128_p8(poly8x16_t a) {
  return vreinterpretq_p128_p8(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_p16
// CHECK: ret
poly128_t test_vreinterpretq_p128_p16(poly16x8_t a) {
  return vreinterpretq_p128_p16(a);
}

// CHECK-LABEL: test_vreinterpretq_p128_p64
// CHECK: ret
poly128_t test_vreinterpretq_p128_p64(poly64x2_t a) {
  return vreinterpretq_p128_p64(a);
}

// CHECK-LABEL: test_vreinterpretq_s8_p128
// CHECK: ret
int8x16_t test_vreinterpretq_s8_p128(poly128_t a) {
  return vreinterpretq_s8_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_s16_p128
// CHECK: ret
int16x8_t test_vreinterpretq_s16_p128(poly128_t  a) {
  return vreinterpretq_s16_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_s32_p128
// CHECK: ret
int32x4_t test_vreinterpretq_s32_p128(poly128_t a) {
  return vreinterpretq_s32_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_s64_p128
// CHECK: ret
int64x2_t test_vreinterpretq_s64_p128(poly128_t  a) {
  return vreinterpretq_s64_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_u8_p128
// CHECK: ret
uint8x16_t test_vreinterpretq_u8_p128(poly128_t  a) {
  return vreinterpretq_u8_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_u16_p128
// CHECK: ret
uint16x8_t test_vreinterpretq_u16_p128(poly128_t  a) {
  return vreinterpretq_u16_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_u32_p128
// CHECK: ret
uint32x4_t test_vreinterpretq_u32_p128(poly128_t  a) {
  return vreinterpretq_u32_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_u64_p128
// CHECK: ret
uint64x2_t test_vreinterpretq_u64_p128(poly128_t  a) {
  return vreinterpretq_u64_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_f32_p128
// CHECK: ret
float32x4_t test_vreinterpretq_f32_p128(poly128_t  a) {
  return vreinterpretq_f32_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_f64_p128
// CHECK: ret
float64x2_t test_vreinterpretq_f64_p128(poly128_t  a) {
  return vreinterpretq_f64_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_p8_p128
// CHECK: ret
poly8x16_t test_vreinterpretq_p8_p128(poly128_t  a) {
  return vreinterpretq_p8_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_p16_p128
// CHECK: ret
poly16x8_t test_vreinterpretq_p16_p128(poly128_t  a) {
  return vreinterpretq_p16_p128(a);
}

// CHECK-LABEL: test_vreinterpretq_p64_p128
// CHECK: ret
poly64x2_t test_vreinterpretq_p64_p128(poly128_t  a) {
  return vreinterpretq_p64_p128(a);
}


