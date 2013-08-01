// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int8x8_t test_vadd_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK: test_vadd_s8
  return vadd_s8(v1, v2);
  // CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vadd_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK: test_vadd_s16
  return vadd_s16(v1, v2);
  // CHECK: add {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vadd_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK: test_vadd_s32
  return vadd_s32(v1, v2);
  // CHECK: add {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vadd_s64(int64x1_t v1, int64x1_t v2) {
  // CHECK: test_vadd_s64
  return vadd_s64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32x2_t test_vadd_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK: test_vadd_f32
  return vadd_f32(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vadd_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK: test_vadd_u8
  return vadd_u8(v1, v2);
  // CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vadd_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK: test_vadd_u16
  return vadd_u16(v1, v2);
  // CHECK: add {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vadd_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK: test_vadd_u32
  return vadd_u32(v1, v2);
  // CHECK: add {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vadd_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK: test_vadd_u64
  return vadd_u64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vaddq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK: test_vaddq_s8
  return vaddq_s8(v1, v2);
  // CHECK: add {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vaddq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK: test_vaddq_s16
  return vaddq_s16(v1, v2);
  // CHECK: add {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vaddq_s32(int32x4_t v1,int32x4_t  v2) {
   // CHECK: test_vaddq_s32
  return vaddq_s32(v1, v2);
  // CHECK: add {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vaddq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK: test_vaddq_s64
  return vaddq_s64(v1, v2);
  // CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vaddq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK: test_vaddq_f32
  return vaddq_f32(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vaddq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vaddq_f64
  return vaddq_f64(v1, v2);
  // CHECK: fadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vaddq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK: test_vaddq_u8
  return vaddq_u8(v1, v2);
  // CHECK: add {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vaddq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK: test_vaddq_u16
  return vaddq_u16(v1, v2);
  // CHECK: add {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vaddq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK: vaddq_u32
  return vaddq_u32(v1, v2);
  // CHECK: add {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vaddq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK: test_vaddq_u64
  return vaddq_u64(v1, v2);
  // CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vsub_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK: test_vsub_s8
  return vsub_s8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}
int16x4_t test_vsub_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK: test_vsub_s16
  return vsub_s16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}
int32x2_t test_vsub_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK: test_vsub_s32
  return vsub_s32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vsub_s64(int64x1_t v1, int64x1_t v2) {
   // CHECK: test_vsub_s64
  return vsub_s64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK: test_vsub_f32
  return vsub_f32(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vsub_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK: test_vsub_u8
  return vsub_u8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vsub_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK: test_vsub_u16
  return vsub_u16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vsub_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK: test_vsub_u32
  return vsub_u32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vsub_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK: test_vsub_u64
  return vsub_u64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vsubq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK: test_vsubq_s8
  return vsubq_s8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vsubq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK: test_vsubq_s16
  return vsubq_s16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vsubq_s32(int32x4_t v1,int32x4_t  v2) {
   // CHECK: test_vsubq_s32
  return vsubq_s32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vsubq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK: test_vsubq_s64
  return vsubq_s64(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK: test_vsubq_f32
  return vsubq_f32(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vsubq_f64
  return vsubq_f64(v1, v2);
  // CHECK: fsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vsubq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK: test_vsubq_u8
  return vsubq_u8(v1, v2);
  // CHECK: sub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vsubq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK: test_vsubq_u16
  return vsubq_u16(v1, v2);
  // CHECK: sub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vsubq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK: vsubq_u32
  return vsubq_u32(v1, v2);
  // CHECK: sub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vsubq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK: test_vsubq_u64
  return vsubq_u64(v1, v2);
  // CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vmul_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vmul_s8
  return vmul_s8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmul_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vmul_s16
  return vmul_s16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmul_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vmul_s32
  return vmul_s32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmul_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vmul_f32
  return vmul_f32(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}


uint8x8_t test_vmul_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vmul_u8
  return vmul_u8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmul_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vmul_u16
  return vmul_u16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmul_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vmul_u32
  return vmul_u32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmulq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vmulq_s8
  return vmulq_s8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmulq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vmulq_s16
  return vmulq_s16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmulq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vmulq_s32
  return vmulq_s32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
    
uint8x16_t test_vmulq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vmulq_u8
  return vmulq_u8(v1, v2);
  // CHECK: mul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmulq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vmulq_u16
  return vmulq_u16(v1, v2);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmulq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vmulq_u32
  return vmulq_u32(v1, v2);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmulq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vmulq_f32
  return vmulq_f32(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmulq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vmulq_f64
  return vmulq_f64(v1, v2);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly8x8_t test_vmul_p8(poly8x8_t v1, poly8x8_t v2) {
  //  test_vmul_p8
  return vmul_p8(v1, v2);
  //  pmul {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x16_t test_vmulq_p8(poly8x16_t v1, poly8x16_t v2) {
  // test_vmulq_p8
  return vmulq_p8(v1, v2);
  // pmul {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}


int8x8_t test_vmla_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK: test_vmla_s8
  return vmla_s8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vmla_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK: test_vmla_s16
  return vmla_s16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmla_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK: test_vmla_s32
  return vmla_s32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmla_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK: test_vmla_f32
  return vmla_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmla_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK: test_vmla_u8
  return vmla_u8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmla_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK: test_vmla_u16
  return vmla_u16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmla_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK: test_vmla_u32
  return vmla_u32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmlaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK: test_vmlaq_s8
  return vmlaq_s8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmlaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK: test_vmlaq_s16
  return vmlaq_s16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmlaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK: test_vmlaq_s32
  return vmlaq_s32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
} 

float32x4_t test_vmlaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK: test_vmlaq_f32
  return vmlaq_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vmlaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
   // CHECK: test_vmlaq_u8
  return vmlaq_u8(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmlaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK: test_vmlaq_u16
  return vmlaq_u16(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmlaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK: test_vmlaq_u32
  return vmlaq_u32(v1, v2, v3);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmlaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: test_vmlaq_f64
  return vmlaq_f64(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vmls_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK: test_vmls_s8
  return vmls_s8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vmls_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK: test_vmls_s16
  return vmls_s16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmls_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK: test_vmls_s32
  return vmls_s32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmls_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK: test_vmls_f32
  return vmls_f32(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmls_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK: test_vmls_u8
  return vmls_u8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmls_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK: test_vmls_u16
  return vmls_u16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmls_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK: test_vmls_u32
  return vmls_u32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}
int8x16_t test_vmlsq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK: test_vmlsq_s8
  return vmlsq_s8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmlsq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK: test_vmlsq_s16
  return vmlsq_s16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmlsq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK: test_vmlsq_s32
  return vmlsq_s32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmlsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK: test_vmlsq_f32
  return vmlsq_f32(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
uint8x16_t test_vmlsq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK: test_vmlsq_u8
  return vmlsq_u8(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmlsq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK: test_vmlsq_u16
  return vmlsq_u16(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmlsq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK: test_vmlsq_u32
  return vmlsq_u32(v1, v2, v3);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmlsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: test_vmlsq_f64
  return vmlsq_f64(v1, v2, v3);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
float32x2_t test_vfma_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK: test_vfma_f32
  return vfma_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vfmaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK: test_vfmaq_f32
  return vfmaq_f32(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vfmaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: test_vfmaq_f64
  return vfmaq_f64(v1, v2, v3);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
float32x2_t test_vfms_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK: test_vfms_f32
  return vfms_f32(v1, v2, v3);
  // CHECK: fmls v0.2s, v1.2s, v2.2s
}

float32x4_t test_vfmsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK: test_vfmsq_f32
  return vfmsq_f32(v1, v2, v3);
  // CHECK: fmls v0.4s, v1.4s, v2.4s
}

float64x2_t test_vfmsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: vfmsq_f64
  return vfmsq_f64(v1, v2, v3);
  // CHECK: fmls v0.2d, v1.2d, v2.2d
}

float64x2_t test_vdivq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vdivq_f64
  return vdivq_f64(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vdivq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vdivq_f32
  return vdivq_f32(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x2_t test_vdiv_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vdiv_f32
  return vdiv_f32(v1, v2);
  // CHECK: fdiv {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vaddd_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK: test_vaddd_u64
  return vaddd_u64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int64x1_t test_vaddd_s64(int64x1_t v1, int64x1_t v2) {
   // CHECK: test_vaddd_s64
  return vaddd_s64(v1, v2);
  // CHECK: add {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x1_t test_vsubd_u64(uint64x1_t v1, uint64x1_t v2) {
   // CHECK: test_vsubd_u64
  return vsubd_u64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int64x1_t test_vsubd_s64(int64x1_t v1, int64x1_t v2) {
   // CHECK: test_vsubd_s64
  return vsubd_s64(v1, v2);
  // CHECK: sub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x8_t test_vaba_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK: test_vaba_s8
  return vaba_s8(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vaba_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK: test_vaba_s16
  return vaba_s16(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vaba_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK: test_vaba_s32
  return vaba_s32(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vaba_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK: test_vaba_u8
  return vaba_u8(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vaba_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK: test_vaba_u16
  return vaba_u16(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vaba_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK: test_vaba_u32
  return vaba_u32(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vabaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK: test_vabaq_s8
  return vabaq_s8(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vabaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK: test_vabaq_s16
  return vabaq_s16(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vabaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK: test_vabaq_s32
  return vabaq_s32(v1, v2, v3);
  // CHECK: saba {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vabaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK: test_vabaq_u8
  return vabaq_u8(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vabaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK: test_vabaq_u16
  return vabaq_u16(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vabaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  // CHECK: test_vabaq_u32
  return vabaq_u32(v1, v2, v3);
  // CHECK: uaba {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int8x8_t test_vabd_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vabd_s8
  return vabd_s8(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vabd_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vabd_s16
  return vabd_s16(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vabd_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vabd_s32
  return vabd_s32(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vabd_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vabd_u8
  return vabd_u8(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vabd_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vabd_u16
  return vabd_u16(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vabd_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vabd_u32
  return vabd_u32(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vabd_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vabd_f32
  return vabd_f32(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vabdq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vabdq_s8
  return vabdq_s8(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vabdq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vabdq_s16
  return vabdq_s16(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vabdq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vabdq_s32
  return vabdq_s32(v1, v2);
  // CHECK: sabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vabdq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vabdq_u8
  return vabdq_u8(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vabdq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vabdq_u16
  return vabdq_u16(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vabdq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vabdq_u32
  return vabdq_u32(v1, v2);
  // CHECK: uabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vabdq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vabdq_f32
  return vabdq_f32(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vabdq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vabdq_f64
  return vabdq_f64(v1, v2);
  // CHECK: fabd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vbsl_s8(uint8x8_t v1, int8x8_t v2, int8x8_t v3) {
  // CHECK: test_vbsl_s8
  return vbsl_s8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x8_t test_vbsl_s16(uint16x4_t v1, int16x4_t v2, int16x4_t v3) {
  // CHECK: test_vbsl_s16
  return vbsl_s16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int32x2_t test_vbsl_s32(uint32x2_t v1, int32x2_t v2, int32x2_t v3) {
  // CHECK: test_vbsl_s32
  return vbsl_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x1_t test_vbsl_s64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  // CHECK: test_vbsl_s64
  return vbsl_s64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x8_t test_vbsl_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  // CHECK: test_vbsl_u8
  return vbsl_u8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vbsl_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  // CHECK: test_vbsl_u16
  return vbsl_u16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint32x2_t test_vbsl_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  // CHECK: test_vbsl_u32
  return vbsl_u32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint64x1_t test_vbsl_u64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  // CHECK: test_vbsl_u64
  return vbsl_u64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

float32x2_t test_vbsl_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  // CHECK: test_vbsl_f32
  return vbsl_f32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly8x8_t test_vbsl_p8(uint8x8_t v1, poly8x8_t v2, poly8x8_t v3) {
  // CHECK: test_vbsl_p8
  return vbsl_p8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly16x4_t test_vbsl_p16(uint16x4_t v1, poly16x4_t v2, poly16x4_t v3) {
  // CHECK: test_vbsl_p16
  return vbsl_p16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int8x16_t test_vbslq_s8(uint8x16_t v1, int8x16_t v2, int8x16_t v3) {
  // CHECK: test_vbslq_s8
  return vbslq_s8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vbslq_s16(uint16x8_t v1, int16x8_t v2, int16x8_t v3) {
  // CHECK: test_vbslq_s16
  return vbslq_s16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vbslq_s32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK: test_vbslq_s32
  return vbslq_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int64x2_t test_vbslq_s64(uint64x2_t v1, int64x2_t v2, int64x2_t v3) {
  // CHECK: test_vbslq_s64
  return vbslq_s64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint8x16_t test_vbslq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  // CHECK: test_vbslq_u8
  return vbslq_u8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vbslq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  // CHECK: test_vbslq_u16
  return vbslq_u16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int32x4_t test_vbslq_u32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  // CHECK: test_vbslq_u32
  return vbslq_s32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint64x2_t test_vbslq_u64(uint64x2_t v1, uint64x2_t v2, uint64x2_t v3) {
  // CHECK: test_vbslq_u64
  return vbslq_u64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float32x4_t test_vbslq_f32(uint32x4_t v1, float32x4_t v2, float32x4_t v3) {
  // CHECK: test_vbslq_f32
  return vbslq_f32(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly8x16_t test_vbslq_p8(uint8x16_t v1, poly8x16_t v2, poly8x16_t v3) {
  // CHECK: test_vbslq_p8
  return vbslq_p8(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly16x8_t test_vbslq_p16(uint16x8_t v1, poly16x8_t v2, poly16x8_t v3) {
  // CHECK: test_vbslq_p16
  return vbslq_p16(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float64x2_t test_vbslq_f64(uint64x2_t v1, float64x2_t v2, float64x2_t v3) {
  // CHECK: test_vbslq_f64
  return vbslq_f64(v1, v2, v3);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

float32x2_t test_vrecps_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK: test_vrecps_f32
   return vrecps_f32(v1, v2);
   // CHECK: frecps {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vrecpsq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK: test_vrecpsq_f32
   return vrecpsq_f32(v1, v2);
   // CHECK: frecps {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vrecpsq_f64(float64x2_t v1, float64x2_t v2) {
   // CHECK: test_vrecpsq_f64
  return vrecpsq_f64(v1, v2);
  // CHECK: frecps {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vrsqrts_f32(float32x2_t v1, float32x2_t v2) {
   // CHECK: test_vrsqrts_f32
  return vrsqrts_f32(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vrsqrtsq_f32(float32x4_t v1, float32x4_t v2) {
   // CHECK: test_vrsqrtsq_f32
  return vrsqrtsq_f32(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vrsqrtsq_f64(float64x2_t v1, float64x2_t v2) {
   // CHECK: test_vrsqrtsq_f64
  return vrsqrtsq_f64(v1, v2);
  // CHECK: frsqrts {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcage_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcage_f32
  return vcage_f32(v1, v2);
  // CHECK: facge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x4_t test_vcageq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcageq_f32
  return vcageq_f32(v1, v2);
  // CHECK: facge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcageq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcageq_f64
  return vcageq_f64(v1, v2);
  // CHECK: facge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcagt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcagt_f32
  return vcagt_f32(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x4_t test_vcagtq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcagtq_f32
  return vcagtq_f32(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcagtq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcagtq_f64
  return vcagtq_f64(v1, v2);
  // CHECK: facgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint32x2_t test_vcale_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcale_f32
  return vcale_f32(v1, v2);
 // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint32x4_t test_vcaleq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcaleq_f32
  return vcaleq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcaleq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcaleq_f64
  return vcaleq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facge {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint32x2_t test_vcalt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcalt_f32
  return vcalt_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint32x4_t test_vcaltq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcaltq_f32
  return vcaltq_f32(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcaltq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcaltq_f64
  return vcaltq_f64(v1, v2);
  // Using registers other than v0, v1 are possible, but would be odd.
  // CHECK: facgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint8x8_t test_vtst_s8(int8x8_t v1, int8x8_t v2) {
   // CHECK: test_vtst_s8
  return vtst_s8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vtst_s16(int16x4_t v1, int16x4_t v2) {
   // CHECK: test_vtst_s16
  return vtst_s16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vtst_s32(int32x2_t v1, int32x2_t v2) {
   // CHECK: test_vtst_s32
  return vtst_s32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vtst_u8(uint8x8_t v1, uint8x8_t v2) {
   // CHECK: test_vtst_u8
  return vtst_u8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vtst_u16(uint16x4_t v1, uint16x4_t v2) {
   // CHECK: test_vtst_u16
  return vtst_u16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vtst_u32(uint32x2_t v1, uint32x2_t v2) {
   // CHECK: test_vtst_u32
  return vtst_u32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vtstq_s8(int8x16_t v1, int8x16_t v2) {
   // CHECK: test_vtstq_s8
  return vtstq_s8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vtstq_s16(int16x8_t v1, int16x8_t v2) {
   // CHECK: test_vtstq_s16
  return vtstq_s16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vtstq_s32(int32x4_t v1, int32x4_t v2) {
   // CHECK: test_vtstq_s32
  return vtstq_s32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vtstq_u8(uint8x16_t v1, uint8x16_t v2) {
   // CHECK: test_vtstq_u8
  return vtstq_u8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vtstq_u16(uint16x8_t v1, uint16x8_t v2) {
   // CHECK: test_vtstq_u16
  return vtstq_u16(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vtstq_u32(uint32x4_t v1, uint32x4_t v2) {
   // CHECK: test_vtstq_u32
  return vtstq_u32(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vtstq_s64(int64x2_t v1, int64x2_t v2) {
   // CHECK: test_vtstq_s64
  return vtstq_s64(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vtstq_u64(uint64x2_t v1, uint64x2_t v2) {
   // CHECK: test_vtstq_u64
  return vtstq_u64(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x8_t test_vtst_p8(poly8x8_t v1, poly8x8_t v2) {
   // CHECK: test_vtst_p8
  return vtst_p8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vtstq_p8(poly8x16_t v1, poly8x16_t v2) {
   // CHECK: test_vtstq_p8
  return vtstq_p8(v1, v2);
  // CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}


uint8x8_t test_vceq_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vceq_s8
  return vceq_s8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vceq_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vceq_s16
  return vceq_s16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vceq_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vceq_s32
  return vceq_s32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x2_t test_vceq_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vceq_f32
  return vceq_f32(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vceq_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vceq_u8
  return vceq_u8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vceq_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vceq_u16
  return vceq_u16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vceq_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vceq_u32
  return vceq_u32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vceq_p8(poly8x8_t v1, poly8x8_t v2) {
  // CHECK: test_vceq_p8
  return vceq_p8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint8x16_t test_vceqq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vceqq_s8
  return vceqq_s8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vceqq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vceqq_s16
  return vceqq_s16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vceqq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vceqq_s32
  return vceqq_s32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vceqq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vceqq_f32
  return vceqq_f32(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vceqq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vceqq_u8
  return vceqq_u8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vceqq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vceqq_u16
  return vceqq_u16(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vceqq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vceqq_u32
  return vceqq_u32(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vceqq_p8(poly8x16_t v1, poly8x16_t v2) {
  // CHECK: test_vceqq_p8
  return vceqq_p8(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}


uint64x2_t test_vceqq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK: test_vceqq_s64
  return vceqq_s64(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vceqq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK: test_vceqq_u64
  return vceqq_u64(v1, v2);
  // CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vceqq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vceqq_f64
  return vceqq_f64(v1, v2);
  // CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}
uint8x8_t test_vcge_s8(int8x8_t v1, int8x8_t v2) {
// CHECK: test_vcge_s8
  return vcge_s8(v1, v2);
// CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcge_s16(int16x4_t v1, int16x4_t v2) {
// CHECK: test_vcge_s16
  return vcge_s16(v1, v2);
// CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcge_s32(int32x2_t v1, int32x2_t v2) {
// CHECK: test_vcge_s32
  return vcge_s32(v1, v2);
// CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x2_t test_vcge_f32(float32x2_t v1, float32x2_t v2) {
// CHECK: test_vcge_f32
  return vcge_f32(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vcge_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK: test_vcge_u8
  return vcge_u8(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcge_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK: test_vcge_u16
  return vcge_u16(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcge_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK: test_vcge_u32
  return vcge_u32(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vcgeq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK: test_vcgeq_s8
  return vcgeq_s8(v1, v2);
// CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgeq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK: test_vcgeq_s16
  return vcgeq_s16(v1, v2);
// CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgeq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK: test_vcgeq_s32
  return vcgeq_s32(v1, v2);
// CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vcgeq_f32(float32x4_t v1, float32x4_t v2) {
// CHECK: test_vcgeq_f32
  return vcgeq_f32(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vcgeq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK: test_vcgeq_u8
  return vcgeq_u8(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgeq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK: test_vcgeq_u16
  return vcgeq_u16(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgeq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK: test_vcgeq_u32
  return vcgeq_u32(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcgeq_s64(int64x2_t v1, int64x2_t v2) {
// CHECK: test_vcgeq_s64
  return vcgeq_s64(v1, v2);
// CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgeq_u64(uint64x2_t v1, uint64x2_t v2) {
// CHECK: test_vcgeq_u64
  return vcgeq_u64(v1, v2);
// CHECK: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgeq_f64(float64x2_t v1, float64x2_t v2) {
// CHECK: test_vcgeq_f64
  return vcgeq_f64(v1, v2);
// CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

// Notes about vcle:
// LE condition predicate implemented as GE, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
uint8x8_t test_vcle_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vcle_s8
  return vcle_s8(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vcle_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vcle_s16
  return vcle_s16(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vcle_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vcle_s32
  return vcle_s32(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint32x2_t test_vcle_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcle_f32
  return vcle_f32(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x8_t test_vcle_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vcle_u8
  return vcle_u8(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vcle_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vcle_u16
  return vcle_u16(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vcle_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vcle_u32
  return vcle_u32(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x16_t test_vcleq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vcleq_s8
  return vcleq_s8(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcleq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vcleq_s16
  return vcleq_s16(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcleq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vcleq_s32
  return vcleq_s32(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint32x4_t test_vcleq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcleq_f32
  return vcleq_f32(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint8x16_t test_vcleq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vcleq_u8
  return vcleq_u8(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcleq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vcleq_u16
  return vcleq_u16(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcleq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vcleq_u32
  return vcleq_u32(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcleq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK: test_vcleq_s64
  return vcleq_s64(v1, v2);
  // CHECK: cmge {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcleq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK: test_vcleq_u64
  return vcleq_u64(v1, v2);
  // CHECK: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcleq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcleq_f64
  return vcleq_f64(v1, v2);
  // CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
}


uint8x8_t test_vcgt_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vcgt_s8
  return vcgt_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcgt_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vcgt_s16
  return vcgt_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcgt_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vcgt_s32
  return vcgt_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint32x2_t test_vcgt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vcgt_f32
  return vcgt_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vcgt_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vcgt_u8
  return vcgt_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vcgt_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vcgt_u16
  return vcgt_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vcgt_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vcgt_u32
  return vcgt_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x16_t test_vcgtq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vcgtq_s8
  return vcgtq_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgtq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vcgtq_s16
  return vcgtq_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgtq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vcgtq_s32
  return vcgtq_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint32x4_t test_vcgtq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcgtq_f32
  return vcgtq_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vcgtq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vcgtq_u8
  return vcgtq_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vcgtq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vcgtq_u16
  return vcgtq_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vcgtq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vcgtq_u32
  return vcgtq_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vcgtq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK: test_vcgtq_s64
  return vcgtq_s64(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgtq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK: test_vcgtq_u64
  return vcgtq_u64(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x2_t test_vcgtq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcgtq_f64
  return vcgtq_f64(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


// Notes about vclt:
// LT condition predicate implemented as GT, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.

uint8x8_t test_vclt_s8(int8x8_t v1, int8x8_t v2) {
  // CHECK: test_vclt_s8
  return vclt_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vclt_s16(int16x4_t v1, int16x4_t v2) {
  // CHECK: test_vclt_s16
  return vclt_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vclt_s32(int32x2_t v1, int32x2_t v2) {
  // CHECK: test_vclt_s32
  return vclt_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint32x2_t test_vclt_f32(float32x2_t v1, float32x2_t v2) {
  // CHECK: test_vclt_f32
  return vclt_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x8_t test_vclt_u8(uint8x8_t v1, uint8x8_t v2) {
  // CHECK: test_vclt_u8
  return vclt_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8b, v1.8b, v0.8b
}

uint16x4_t test_vclt_u16(uint16x4_t v1, uint16x4_t v2) {
  // CHECK: test_vclt_u16
  return vclt_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
}

uint32x2_t test_vclt_u32(uint32x2_t v1, uint32x2_t v2) {
  // CHECK: test_vclt_u32
  return vclt_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
}

uint8x16_t test_vcltq_s8(int8x16_t v1, int8x16_t v2) {
  // CHECK: test_vcltq_s8
  return vcltq_s8(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcltq_s16(int16x8_t v1, int16x8_t v2) {
  // CHECK: test_vcltq_s16
  return vcltq_s16(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcltq_s32(int32x4_t v1, int32x4_t v2) {
  // CHECK: test_vcltq_s32
  return vcltq_s32(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint32x4_t test_vcltq_f32(float32x4_t v1, float32x4_t v2) {
  // CHECK: test_vcltq_f32
  return vcltq_f32(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint8x16_t test_vcltq_u8(uint8x16_t v1, uint8x16_t v2) {
  // CHECK: test_vcltq_u8
  return vcltq_u8(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
}

uint16x8_t test_vcltq_u16(uint16x8_t v1, uint16x8_t v2) {
  // CHECK: test_vcltq_u16
  return vcltq_u16(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
}

uint32x4_t test_vcltq_u32(uint32x4_t v1, uint32x4_t v2) {
  // CHECK: test_vcltq_u32
  return vcltq_u32(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
}

uint64x2_t test_vcltq_s64(int64x2_t v1, int64x2_t v2) {
  // CHECK: test_vcltq_s64
  return vcltq_s64(v1, v2);
  // CHECK: cmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcltq_u64(uint64x2_t v1, uint64x2_t v2) {
  // CHECK: test_vcltq_u64
  return vcltq_u64(v1, v2);
  // CHECK: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
}

uint64x2_t test_vcltq_f64(float64x2_t v1, float64x2_t v2) {
  // CHECK: test_vcltq_f64
  return vcltq_f64(v1, v2);
  // CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
}


int8x8_t test_vhadd_s8(int8x8_t v1, int8x8_t v2) {
// CHECK: test_vhadd_s8
  return vhadd_s8(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vhadd_s16(int16x4_t v1, int16x4_t v2) {
// CHECK: test_vhadd_s16
  return vhadd_s16(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vhadd_s32(int32x2_t v1, int32x2_t v2) {
// CHECK: test_vhadd_s32
  return vhadd_s32(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vhadd_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK: test_vhadd_u8
  return vhadd_u8(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vhadd_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK: test_vhadd_u16
  return vhadd_u16(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vhadd_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK: test_vhadd_u32
  return vhadd_u32(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vhaddq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK: test_vhaddq_s8
  return vhaddq_s8(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vhaddq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK: test_vhaddq_s16
  return vhaddq_s16(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vhaddq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK: test_vhaddq_s32
  return vhaddq_s32(v1, v2);
  // CHECK: shadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK: test_vhaddq_u8
  return vhaddq_u8(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK: test_vhaddq_u16
  return vhaddq_u16(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK: test_vhaddq_u32
  return vhaddq_u32(v1, v2);
  // CHECK: uhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}


int8x8_t test_vhsub_s8(int8x8_t v1, int8x8_t v2) {
// CHECK: test_vhsub_s8
  return vhsub_s8(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vhsub_s16(int16x4_t v1, int16x4_t v2) {
// CHECK: test_vhsub_s16
  return vhsub_s16(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vhsub_s32(int32x2_t v1, int32x2_t v2) {
// CHECK: test_vhsub_s32
  return vhsub_s32(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vhsub_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK: test_vhsub_u8
  return vhsub_u8(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vhsub_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK: test_vhsub_u16
  return vhsub_u16(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vhsub_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK: test_vhsub_u32
  return vhsub_u32(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vhsubq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK: test_vhsubq_s8
  return vhsubq_s8(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vhsubq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK: test_vhsubq_s16
  return vhsubq_s16(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vhsubq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK: test_vhsubq_s32
  return vhsubq_s32(v1, v2);
  // CHECK: shsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vhsubq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK: test_vhsubq_u8
  return vhsubq_u8(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vhsubq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK: test_vhsubq_u16
  return vhsubq_u16(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vhsubq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK: test_vhsubq_u32
  return vhsubq_u32(v1, v2);
  // CHECK: uhsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}


int8x8_t test_vrhadd_s8(int8x8_t v1, int8x8_t v2) {
// CHECK: test_vrhadd_s8
  return vrhadd_s8(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vrhadd_s16(int16x4_t v1, int16x4_t v2) {
// CHECK: test_vrhadd_s16
  return vrhadd_s16(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vrhadd_s32(int32x2_t v1, int32x2_t v2) {
// CHECK: test_vrhadd_s32
  return vrhadd_s32(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vrhadd_u8(uint8x8_t v1, uint8x8_t v2) {
// CHECK: test_vrhadd_u8
  return vrhadd_u8(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vrhadd_u16(uint16x4_t v1, uint16x4_t v2) {
// CHECK: test_vrhadd_u16
  return vrhadd_u16(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vrhadd_u32(uint32x2_t v1, uint32x2_t v2) {
// CHECK: test_vrhadd_u32
  return vrhadd_u32(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vrhaddq_s8(int8x16_t v1, int8x16_t v2) {
// CHECK: test_vrhaddq_s8
  return vrhaddq_s8(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vrhaddq_s16(int16x8_t v1, int16x8_t v2) {
// CHECK: test_vrhaddq_s16
  return vrhaddq_s16(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vrhaddq_s32(int32x4_t v1, int32x4_t v2) {
// CHECK: test_vrhaddq_s32
  return vrhaddq_s32(v1, v2);
// CHECK: srhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vrhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
// CHECK: test_vrhaddq_u8
  return vrhaddq_u8(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vrhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
// CHECK: test_vrhaddq_u16
  return vrhaddq_u16(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vrhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
// CHECK: test_vrhaddq_u32
  return vrhaddq_u32(v1, v2);
// CHECK: urhadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}
int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vqadd_s8
  return vqadd_s8(a, b);
  // CHECK: sqadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqadd_s16
  return vqadd_s16(a, b);
  // CHECK: sqadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqadd_s32
  return vqadd_s32(a, b);
  // CHECK: sqadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vqadd_s64
  return vqadd_s64(a, b);
// CHECK:  sqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vqadd_u8
  return vqadd_u8(a, b);
  // CHECK: uqadd {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vqadd_u16
  return vqadd_u16(a, b);
  // CHECK: uqadd {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vqadd_u32
  return vqadd_u32(a, b);
  // CHECK: uqadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
// CHECK:  test_vqadd_u64
  return vqadd_u64(a, b);
// CHECK:  uqadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vqaddq_s8
  return vqaddq_s8(a, b);
  // CHECK: sqadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqaddq_s16
  return vqaddq_s16(a, b);
  // CHECK: sqadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqaddq_s32
  return vqaddq_s32(a, b);
  // CHECK: sqadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vqaddq_s64
  return vqaddq_s64(a, b);
// CHECK: sqadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vqaddq_u8
  return vqaddq_u8(a, b);
  // CHECK: uqadd {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vqaddq_u16
  return vqaddq_u16(a, b);
  // CHECK: uqadd {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqaddq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vqaddq_u32
  return vqaddq_u32(a, b);
  // CHECK: uqadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
// CHECK: test_vqaddq_u64
  return vqaddq_u64(a, b);
// CHECK: uqadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vqsub_s8
  return vqsub_s8(a, b);
  // CHECK: sqsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqsub_s16
  return vqsub_s16(a, b);
  // CHECK: sqsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqsub_s32
  return vqsub_s32(a, b);
  // CHECK: sqsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vqsub_s64
  return vqsub_s64(a, b);
// CHECK: sqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vqsub_u8
  return vqsub_u8(a, b);
  // CHECK: uqsub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vqsub_u16
  return vqsub_u16(a, b);
  // CHECK: uqsub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vqsub_u32
  return vqsub_u32(a, b);
  // CHECK: uqsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
// CHECK: test_vqsub_u64
  return vqsub_u64(a, b);
// CHECK:  uqsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vqsubq_s8
  return vqsubq_s8(a, b);
  // CHECK: sqsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqsubq_s16
  return vqsubq_s16(a, b);
  // CHECK: sqsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqsubq_s32
  return vqsubq_s32(a, b);
  // CHECK: sqsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vqsubq_s64
  return vqsubq_s64(a, b);
// CHECK: sqsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vqsubq_u8
  return vqsubq_u8(a, b);
  // CHECK: uqsub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vqsubq_u16
  return vqsubq_u16(a, b);
  // CHECK: uqsub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vqsubq_u32
  return vqsubq_u32(a, b);
  // CHECK: uqsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
// CHECK: test_vqsubq_u64
  return vqsubq_u64(a, b);
  // CHECK: uqsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vshl_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vshl_s8
  return vshl_s8(a, b);
// CHECK: sshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vshl_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vshl_s16
  return vshl_s16(a, b);
// CHECK: sshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vshl_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vshl_s32
  return vshl_s32(a, b);
// CHECK: sshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vshl_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vshl_s64
  return vshl_s64(a, b);
// CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK: test_vshl_u8
  return vshl_u8(a, b);
// CHECK: ushl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK: test_vshl_u16
  return vshl_u16(a, b);
// CHECK: ushl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK: test_vshl_u32
  return vshl_u32(a, b);
// CHECK: ushl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK: test_vshl_u64
  return vshl_u64(a, b);
// CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vshlq_s8
  return vshlq_s8(a, b);
// CHECK: sshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vshlq_s16
  return vshlq_s16(a, b);
// CHECK: sshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vshlq_s32
  return vshlq_s32(a, b);
// CHECK: sshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vshlq_s64
  return vshlq_s64(a, b);
// CHECK: sshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK: test_vshlq_u8
  return vshlq_u8(a, b);
// CHECK: ushl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK: test_vshlq_u16
  return vshlq_u16(a, b);
// CHECK: ushl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK: test_vshlq_u32
  return vshlq_u32(a, b);
// CHECK: ushl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK: test_vshlq_u64
  return vshlq_u64(a, b);
// CHECK: ushl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqshl_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vqshl_s8
  return vqshl_s8(a, b);
// CHECK: sqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqshl_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqshl_s16
  return vqshl_s16(a, b);
// CHECK: sqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqshl_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqshl_s32
  return vqshl_s32(a, b);
// CHECK: sqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqshl_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vqshl_s64
  return vqshl_s64(a, b);
// CHECK: sqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK: test_vqshl_u8
  return vqshl_u8(a, b);
// CHECK: uqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK: test_vqshl_u16
  return vqshl_u16(a, b);
// CHECK: uqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK: test_vqshl_u32
  return vqshl_u32(a, b);
// CHECK: uqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK: test_vqshl_u64
  return vqshl_u64(a, b);
// CHECK: uqshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vqshlq_s8
  return vqshlq_s8(a, b);
// CHECK: sqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqshlq_s16
  return vqshlq_s16(a, b);
// CHECK: sqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqshlq_s32
  return vqshlq_s32(a, b);
// CHECK: sqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vqshlq_s64
  return vqshlq_s64(a, b);
// CHECK: sqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vqshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK: test_vqshlq_u8
  return vqshlq_u8(a, b);
// CHECK: uqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK: test_vqshlq_u16
  return vqshlq_u16(a, b);
// CHECK: uqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK: test_vqshlq_u32
  return vqshlq_u32(a, b);
// CHECK: uqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK: test_vqshlq_u32
  return vqshlq_u64(a, b);
// CHECK: uqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vrshl_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vrshl_s8
  return vrshl_s8(a, b);
// CHECK: srshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vrshl_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vrshl_s16
  return vrshl_s16(a, b);
// CHECK: srshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vrshl_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vrshl_s32
  return vrshl_s32(a, b);
// CHECK: srshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vrshl_s64
  return vrshl_s64(a, b);
// CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vrshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK: test_vrshl_u8
  return vrshl_u8(a, b);
// CHECK: urshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vrshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK: test_vrshl_u16
  return vrshl_u16(a, b);
// CHECK: urshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vrshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK: test_vrshl_u32
  return vrshl_u32(a, b);
// CHECK: urshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK: test_vrshl_u64
  return vrshl_u64(a, b);
// CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vrshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vrshlq_s8
  return vrshlq_s8(a, b);
// CHECK: srshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vrshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vrshlq_s16
  return vrshlq_s16(a, b);
// CHECK: srshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vrshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vrshlq_s32
  return vrshlq_s32(a, b);
// CHECK: srshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vrshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vrshlq_s64
  return vrshlq_s64(a, b);
// CHECK: srshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint8x16_t test_vrshlq_u8(uint8x16_t a, int8x16_t b) {
// CHECK: test_vrshlq_u8
  return vrshlq_u8(a, b);
// CHECK: urshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vrshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK: test_vrshlq_u16
  return vrshlq_u16(a, b);
// CHECK: urshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vrshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK: test_vrshlq_u32
  return vrshlq_u32(a, b);
// CHECK: urshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vrshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK: test_vrshlq_u64
  return vrshlq_u64(a, b);
// CHECK: urshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vqrshl_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vqrshl_s8
  return vqrshl_s8(a, b);
// CHECK: sqrshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vqrshl_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqrshl_s16
  return vqrshl_s16(a, b);
// CHECK: sqrshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqrshl_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqrshl_s32
  return vqrshl_s32(a, b);
// CHECK: sqrshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int64x1_t test_vqrshl_s64(int64x1_t a, int64x1_t b) {
// CHECK: test_vqrshl_s64
  return vqrshl_s64(a, b);
// CHECK: sqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint8x8_t test_vqrshl_u8(uint8x8_t a, int8x8_t b) {
// CHECK: test_vqrshl_u8
  return vqrshl_u8(a, b);
// CHECK: uqrshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vqrshl_u16(uint16x4_t a, int16x4_t b) {
// CHECK: test_vqrshl_u16
  return vqrshl_u16(a, b);
// CHECK: uqrshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vqrshl_u32(uint32x2_t a, int32x2_t b) {
// CHECK: test_vqrshl_u32
  return vqrshl_u32(a, b);
// CHECK: uqrshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint64x1_t test_vqrshl_u64(uint64x1_t a, int64x1_t b) {
// CHECK: test_vqrshl_u64
  return vqrshl_u64(a, b);
// CHECK: uqrshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

int8x16_t test_vqrshlq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vqrshlq_s8
  return vqrshlq_s8(a, b);
// CHECK: sqrshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vqrshlq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqrshlq_s16
  return vqrshlq_s16(a, b);
// CHECK: sqrshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqrshlq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqrshlq_s32
  return vqrshlq_s32(a, b);
// CHECK: sqrshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int64x2_t test_vqrshlq_s64(int64x2_t a, int64x2_t b) {
// CHECK: test_vqrshlq_s64
  return vqrshlq_s64(a, b);
// CHECK: sqrshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

// CHECK: test_vqrshlq_u8
uint8x16_t test_vqrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vqrshlq_u8(a, b);
// CHECK: uqrshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vqrshlq_u16(uint16x8_t a, int16x8_t b) {
// CHECK: test_vqrshlq_u16
  return vqrshlq_u16(a, b);
// CHECK: uqrshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vqrshlq_u32(uint32x4_t a, int32x4_t b) {
// CHECK: test_vqrshlq_u32
  return vqrshlq_u32(a, b);
// CHECK: uqrshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint64x2_t test_vqrshlq_u64(uint64x2_t a, int64x2_t b) {
// CHECK: test_vqrshlq_u64
  return vqrshlq_u64(a, b);
// CHECK: uqrshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vmax_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vmax_s8
  return vmax_s8(a, b);
// CHECK: smax {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmax_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vmax_s16
  return vmax_s16(a, b);
// CHECK: smax {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmax_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vmax_s32
  return vmax_s32(a, b);
// CHECK: smax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmax_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vmax_u8
  return vmax_u8(a, b);
// CHECK: umax {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmax_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vmax_u16
  return vmax_u16(a, b);
// CHECK: umax {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmax_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vmax_u32
  return vmax_u32(a, b);
// CHECK: umax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmax_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vmax_f32
  return vmax_f32(a, b);
// CHECK: fmax {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vmaxq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vmaxq_s8
  return vmaxq_s8(a, b);
// CHECK: smax {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vmaxq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vmaxq_s16
  return vmaxq_s16(a, b);
// CHECK: smax {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vmaxq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vmaxq_s32
  return vmaxq_s32(a, b);
// CHECK: smax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vmaxq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vmaxq_u8
  return vmaxq_u8(a, b);
// CHECK: umax {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vmaxq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vmaxq_u16
  return vmaxq_u16(a, b);
// CHECK: umax {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vmaxq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vmaxq_u32
  return vmaxq_u32(a, b);
// CHECK: umax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vmaxq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vmaxq_f32
  return vmaxq_f32(a, b);
// CHECK: fmax {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmaxq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vmaxq_f64
  return vmaxq_f64(a, b);
// CHECK: fmax {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}


int8x8_t test_vmin_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vmin_s8
  return vmin_s8(a, b);
// CHECK: smin {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vmin_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vmin_s16
  return vmin_s16(a, b);
// CHECK: smin {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vmin_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vmin_s32
  return vmin_s32(a, b);
// CHECK: smin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vmin_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vmin_u8
  return vmin_u8(a, b);
// CHECK: umin {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vmin_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vmin_u16
  return vmin_u16(a, b);
// CHECK: umin {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vmin_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vmin_u32
  return vmin_u32(a, b);
// CHECK: umin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vmin_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vmin_f32
  return vmin_f32(a, b);
// CHECK: fmin {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vminq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vminq_s8
  return vminq_s8(a, b);
// CHECK: smin {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vminq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vminq_s16
  return vminq_s16(a, b);
// CHECK: smin {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vminq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vminq_s32
  return vminq_s32(a, b);
// CHECK: smin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vminq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vminq_u8
  return vminq_u8(a, b);
// CHECK: umin {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vminq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vminq_u16
  return vminq_u16(a, b);
// CHECK: umin {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vminq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vminq_u32
  return vminq_u32(a, b);
// CHECK: umin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vminq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vminq_f32
  return vminq_f32(a, b);
// CHECK: fmin {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vminq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vminq_f64
  return vminq_f64(a, b);
// CHECK: fmin {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vmaxnm_f32
  return vmaxnm_f32(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vmaxnmq_f32
  return vmaxnmq_f32(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmaxnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vmaxnmq_f64
  return vmaxnmq_f64(a, b);
// CHECK: fmaxnm {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vminnm_f32
  return vminnm_f32(a, b);
// CHECK: fminnm {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vminnmq_f32
  return vminnmq_f32(a, b);
// CHECK: fminnm {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vminnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vminnmq_f64
  return vminnmq_f64(a, b);
// CHECK: fminnm {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpmax_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vpmax_s8
  return vpmax_s8(a, b);
// CHECK: smaxp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpmax_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vpmax_s16
  return vpmax_s16(a, b);
// CHECK: smaxp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpmax_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vpmax_s32
  return vpmax_s32(a, b);
// CHECK: smaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpmax_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vpmax_u8
  return vpmax_u8(a, b);
// CHECK: umaxp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpmax_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vpmax_u16
  return vpmax_u16(a, b);
// CHECK: umaxp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpmax_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vpmax_u32
  return vpmax_u32(a, b);
// CHECK: umaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpmax_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vpmax_f32
  return vpmax_f32(a, b);
// CHECK: fmaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpmaxq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vpmaxq_s8
  return vpmaxq_s8(a, b);
// CHECK: smaxp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpmaxq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vpmaxq_s16
  return vpmaxq_s16(a, b);
// CHECK: smaxp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpmaxq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vpmaxq_s32
  return vpmaxq_s32(a, b);
// CHECK: smaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpmaxq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vpmaxq_u8
  return vpmaxq_u8(a, b);
// CHECK: umaxp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpmaxq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vpmaxq_u16
  return vpmaxq_u16(a, b);
// CHECK: umaxp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpmaxq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vpmaxq_u32
  return vpmaxq_u32(a, b);
// CHECK: umaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpmaxq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vpmaxq_f32
  return vpmaxq_f32(a, b);
// CHECK: fmaxp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpmaxq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vpmaxq_f64
  return vpmaxq_f64(a, b);
// CHECK: fmaxp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpmin_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vpmin_s8
  return vpmin_s8(a, b);
// CHECK: sminp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpmin_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vpmin_s16
  return vpmin_s16(a, b);
// CHECK: sminp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpmin_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vpmin_s32
  return vpmin_s32(a, b);
// CHECK: sminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpmin_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vpmin_u8
  return vpmin_u8(a, b);
// CHECK: uminp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpmin_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vpmin_u16
  return vpmin_u16(a, b);
// CHECK: uminp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpmin_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vpmin_u32
  return vpmin_u32(a, b);
// CHECK: uminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpmin_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vpmin_f32
  return vpmin_f32(a, b);
// CHECK: fminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpminq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vpminq_s8
  return vpminq_s8(a, b);
// CHECK: sminp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpminq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vpminq_s16
  return vpminq_s16(a, b);
// CHECK: sminp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpminq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vpminq_s32
  return vpminq_s32(a, b);
// CHECK: sminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpminq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vpminq_u8
  return vpminq_u8(a, b);
// CHECK: uminp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpminq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vpminq_u16
  return vpminq_u16(a, b);
// CHECK: uminp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpminq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vpminq_u32
  return vpminq_u32(a, b);
// CHECK: uminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpminq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vpminq_f32
  return vpminq_f32(a, b);
// CHECK: fminp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpminq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vpminq_f64
  return vpminq_f64(a, b);
// CHECK: fminp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vpmaxnm_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vpmaxnm_f32
  return vpmaxnm_f32(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vpmaxnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vpmaxnmq_f32
  return vpmaxnmq_f32(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpmaxnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vpmaxnmq_f64
  return vpmaxnmq_f64(a, b);
// CHECK: fmaxnmp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vpminnm_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vpminnm_f32
  return vpminnm_f32(a, b);
// CHECK: fminnmp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vpminnmq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vpminnmq_f32
  return vpminnmq_f32(a, b);
// CHECK: fminnmp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpminnmq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vpminnmq_f64
  return vpminnmq_f64(a, b);
// CHECK: fminnmp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
// CHECK: test_vpadd_s8
  return vpadd_s8(a, b);
// CHECK: addp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vpadd_s16
  return vpadd_s16(a, b);
// CHECK: addp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vpadd_s32
  return vpadd_s32(a, b);
// CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
// CHECK: test_vpadd_u8
  return vpadd_u8(a, b);
// CHECK: addp {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
// CHECK: test_vpadd_u16
  return vpadd_u16(a, b);
// CHECK: addp {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

uint32x2_t test_vpadd_u32(uint32x2_t a, uint32x2_t b) {
// CHECK: test_vpadd_u32
  return vpadd_u32(a, b);
// CHECK: addp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vpadd_f32
  return vpadd_f32(a, b);
// CHECK: faddp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int8x16_t test_vpaddq_s8(int8x16_t a, int8x16_t b) {
// CHECK: test_vpaddq_s8
  return vpaddq_s8(a, b);
// CHECK: addp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

int16x8_t test_vpaddq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vpaddq_s16
  return vpaddq_s16(a, b);
// CHECK: addp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vpaddq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vpaddq_s32
  return vpaddq_s32(a, b);
// CHECK: addp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

uint8x16_t test_vpaddq_u8(uint8x16_t a, uint8x16_t b) {
// CHECK: test_vpaddq_u8
  return vpaddq_u8(a, b);
// CHECK: addp {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

uint16x8_t test_vpaddq_u16(uint16x8_t a, uint16x8_t b) {
// CHECK: test_vpaddq_u16
  return vpaddq_u16(a, b);
// CHECK: addp {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

uint32x4_t test_vpaddq_u32(uint32x4_t a, uint32x4_t b) {
// CHECK: test_vpaddq_u32
  return vpaddq_u32(a, b);
// CHECK: addp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x4_t test_vpaddq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vpaddq_f32
  return vpaddq_f32(a, b);
// CHECK: faddp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vpaddq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vpaddq_f64
  return vpaddq_f64(a, b);
// CHECK: faddp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

int16x4_t test_vqdmulh_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqdmulh_s16
  return vqdmulh_s16(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqdmulh_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqdmulh_s32
  return vqdmulh_s32(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vqdmulhq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqdmulhq_s16
  return vqdmulhq_s16(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqdmulhq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqdmulhq_s32
  return vqdmulhq_s32(a, b);
// CHECK: sqdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

int16x4_t test_vqrdmulh_s16(int16x4_t a, int16x4_t b) {
// CHECK: test_vqrdmulh_s16
  return vqrdmulh_s16(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
}

int32x2_t test_vqrdmulh_s32(int32x2_t a, int32x2_t b) {
// CHECK: test_vqrdmulh_s32
  return vqrdmulh_s32(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

int16x8_t test_vqrdmulhq_s16(int16x8_t a, int16x8_t b) {
// CHECK: test_vqrdmulhq_s16
  return vqrdmulhq_s16(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
}

int32x4_t test_vqrdmulhq_s32(int32x4_t a, int32x4_t b) {
// CHECK: test_vqrdmulhq_s32
  return vqrdmulhq_s32(a, b);
// CHECK: sqrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}


float32x2_t test_vmulx_f32(float32x2_t a, float32x2_t b) {
// CHECK: test_vmulx_f32
  return vmulx_f32(a, b);
// CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vmulxq_f32(float32x4_t a, float32x4_t b) {
// CHECK: test_vmulxq_f32
  return vmulxq_f32(a, b);
// CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmulxq_f64(float64x2_t a, float64x2_t b) {
// CHECK: test_vmulxq_f64
  return vmulxq_f64(a, b);
// CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

