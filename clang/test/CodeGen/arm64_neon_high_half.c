// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-feature +neon -ffreestanding -Os -S -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

int16x8_t test_vaddw_high_s8(int16x8_t lhs, int8x16_t rhs) {
  // CHECK: saddw2.8h
  return vaddw_high_s8(lhs, rhs);
}

int32x4_t test_vaddw_high_s16(int32x4_t lhs, int16x8_t rhs) {
  // CHECK: saddw2.4s
  return vaddw_high_s16(lhs, rhs);
}

int64x2_t test_vaddw_high_s32(int64x2_t lhs, int32x4_t rhs) {
  // CHECK: saddw2.2d
  return vaddw_high_s32(lhs, rhs);
}

uint16x8_t test_vaddw_high_u8(uint16x8_t lhs, uint8x16_t rhs) {
  // CHECK: uaddw2.8h
  return vaddw_high_u8(lhs, rhs);
}

uint32x4_t test_vaddw_high_u16(uint32x4_t lhs, uint16x8_t rhs) {
  // CHECK: uaddw2.4s
  return vaddw_high_u16(lhs, rhs);
}

uint64x2_t test_vaddw_high_u32(uint64x2_t lhs, uint32x4_t rhs) {
  // CHECK: uaddw2.2d
  return vaddw_high_u32(lhs, rhs);
}

int16x8_t test_vsubw_high_s8(int16x8_t lhs, int8x16_t rhs) {
  // CHECK: ssubw2.8h
  return vsubw_high_s8(lhs, rhs);
}

int32x4_t test_vsubw_high_s16(int32x4_t lhs, int16x8_t rhs) {
  // CHECK: ssubw2.4s
  return vsubw_high_s16(lhs, rhs);
}

int64x2_t test_vsubw_high_s32(int64x2_t lhs, int32x4_t rhs) {
  // CHECK: ssubw2.2d
  return vsubw_high_s32(lhs, rhs);
}

uint16x8_t test_vsubw_high_u8(uint16x8_t lhs, uint8x16_t rhs) {
  // CHECK: usubw2.8h
  return vsubw_high_u8(lhs, rhs);
}

uint32x4_t test_vsubw_high_u16(uint32x4_t lhs, uint16x8_t rhs) {
  // CHECK: usubw2.4s
  return vsubw_high_u16(lhs, rhs);
}

uint64x2_t test_vsubw_high_u32(uint64x2_t lhs, uint32x4_t rhs) {
  // CHECK: usubw2.2d
  return vsubw_high_u32(lhs, rhs);
}

int16x8_t test_vabdl_high_s8(int8x16_t lhs, int8x16_t rhs) {
  // CHECK: sabdl2.8h
  return vabdl_high_s8(lhs, rhs);
}

int32x4_t test_vabdl_high_s16(int16x8_t lhs, int16x8_t rhs) {
  // CHECK: sabdl2.4s
  return vabdl_high_s16(lhs, rhs);
}

int64x2_t test_vabdl_high_s32(int32x4_t lhs, int32x4_t rhs) {
  // CHECK: sabdl2.2d
  return vabdl_high_s32(lhs, rhs);
}

uint16x8_t test_vabdl_high_u8(uint8x16_t lhs, uint8x16_t rhs) {
  // CHECK: uabdl2.8h
  return vabdl_high_u8(lhs, rhs);
}

uint32x4_t test_vabdl_high_u16(uint16x8_t lhs, uint16x8_t rhs) {
  // CHECK: uabdl2.4s
  return vabdl_high_u16(lhs, rhs);
}

uint64x2_t test_vabdl_high_u32(uint32x4_t lhs, uint32x4_t rhs) {
  // CHECK: uabdl2.2d
  return vabdl_high_u32(lhs, rhs);
}

int16x8_t test_vabal_high_s8(int16x8_t accum, int8x16_t lhs, int8x16_t rhs) {
  // CHECK: sabal2.8h
  return vabal_high_s8(accum, lhs, rhs);
}

int32x4_t test_vabal_high_s16(int32x4_t accum, int16x8_t lhs, int16x8_t rhs) {
  // CHECK: sabal2.4s
  return vabal_high_s16(accum, lhs, rhs);
}

int64x2_t test_vabal_high_s32(int64x2_t accum, int32x4_t lhs, int32x4_t rhs) {
  // CHECK: sabal2.2d
  return vabal_high_s32(accum, lhs, rhs);
}

uint16x8_t test_vabal_high_u8(uint16x8_t accum, uint8x16_t lhs, uint8x16_t rhs) {
  // CHECK: uabal2.8h
  return vabal_high_u8(accum, lhs, rhs);
}

uint32x4_t test_vabal_high_u16(uint32x4_t accum, uint16x8_t lhs, uint16x8_t rhs) {
  // CHECK: uabal2.4s
  return vabal_high_u16(accum, lhs, rhs);
}

uint64x2_t test_vabal_high_u32(uint64x2_t accum, uint32x4_t lhs, uint32x4_t rhs) {
  // CHECK: uabal2.2d
  return vabal_high_u32(accum, lhs, rhs);
}

int32x4_t test_vqdmlal_high_s16(int32x4_t accum, int16x8_t lhs, int16x8_t rhs) {
  // CHECK: sqdmlal2.4s
  return vqdmlal_high_s16(accum, lhs, rhs);
}

int64x2_t test_vqdmlal_high_s32(int64x2_t accum, int32x4_t lhs, int32x4_t rhs) {
  // CHECK: sqdmlal2.2d
  return vqdmlal_high_s32(accum, lhs, rhs);
}

int32x4_t test_vqdmlsl_high_s16(int32x4_t accum, int16x8_t lhs, int16x8_t rhs) {
  // CHECK: sqdmlsl2.4s
  return vqdmlsl_high_s16(accum, lhs, rhs);
}

int64x2_t test_vqdmlsl_high_s32(int64x2_t accum, int32x4_t lhs, int32x4_t rhs) {
  // CHECK: sqdmlsl2.2d
  return vqdmlsl_high_s32(accum, lhs, rhs);
}

int32x4_t test_vqdmull_high_s16(int16x8_t lhs, int16x8_t rhs) {
  // CHECK: sqdmull2.4s
  return vqdmull_high_s16(lhs, rhs);
}

int64x2_t test_vqdmull_high_s32(int32x4_t lhs, int32x4_t rhs) {
  // CHECK: sqdmull2.2d
  return vqdmull_high_s32(lhs, rhs);
}

int16x8_t test_vshll_high_n_s8(int8x16_t in) {
  // CHECK: sshll2.8h
  return vshll_high_n_s8(in, 7);
}

int32x4_t test_vshll_high_n_s16(int16x8_t in) {
  // CHECK: sshll2.4s
  return vshll_high_n_s16(in, 15);
}

int64x2_t test_vshll_high_n_s32(int32x4_t in) {
  // CHECK: sshll2.2d
  return vshll_high_n_s32(in, 31);
}

int16x8_t test_vshll_high_n_u8(int8x16_t in) {
  // CHECK: ushll2.8h
  return vshll_high_n_u8(in, 7);
}

int32x4_t test_vshll_high_n_u16(int16x8_t in) {
  // CHECK: ushll2.4s
  return vshll_high_n_u16(in, 15);
}

int64x2_t test_vshll_high_n_u32(int32x4_t in) {
  // CHECK: ushll2.2d
  return vshll_high_n_u32(in, 31);
}

int16x8_t test_vshll_high_n_s8_max(int8x16_t in) {
  // CHECK: shll2.8h
  return vshll_high_n_s8(in, 8);
}

int32x4_t test_vshll_high_n_s16_max(int16x8_t in) {
  // CHECK: shll2.4s
  return vshll_high_n_s16(in, 16);
}

int64x2_t test_vshll_high_n_s32_max(int32x4_t in) {
  // CHECK: shll2.2d
  return vshll_high_n_s32(in, 32);
}

int16x8_t test_vshll_high_n_u8_max(int8x16_t in) {
  // CHECK: shll2.8h
  return vshll_high_n_u8(in, 8);
}

int32x4_t test_vshll_high_n_u16_max(int16x8_t in) {
  // CHECK: shll2.4s
  return vshll_high_n_u16(in, 16);
}

int64x2_t test_vshll_high_n_u32_max(int32x4_t in) {
  // CHECK: shll2.2d
  return vshll_high_n_u32(in, 32);
}

int16x8_t test_vsubl_high_s8(int8x16_t lhs, int8x16_t rhs) {
  // CHECK: ssubl2.8h
  return vsubl_high_s8(lhs, rhs);
}

int32x4_t test_vsubl_high_s16(int16x8_t lhs, int16x8_t rhs) {
  // CHECK: ssubl2.4s
  return vsubl_high_s16(lhs, rhs);
}

int64x2_t test_vsubl_high_s32(int32x4_t lhs, int32x4_t rhs) {
  // CHECK: ssubl2.2d
  return vsubl_high_s32(lhs, rhs);
}

uint16x8_t test_vsubl_high_u8(uint8x16_t lhs, uint8x16_t rhs) {
  // CHECK: usubl2.8h
  return vsubl_high_u8(lhs, rhs);
}

uint32x4_t test_vsubl_high_u16(uint16x8_t lhs, uint16x8_t rhs) {
  // CHECK: usubl2.4s
  return vsubl_high_u16(lhs, rhs);
}

uint64x2_t test_vsubl_high_u32(uint32x4_t lhs, uint32x4_t rhs) {
  // CHECK: usubl2.2d
  return vsubl_high_u32(lhs, rhs);
}

int8x16_t test_vrshrn_high_n_s16(int8x8_t lowpart, int16x8_t input) {
  // CHECK: rshrn2.16b
  return vrshrn_high_n_s16(lowpart, input, 2);
}

int16x8_t test_vrshrn_high_n_s32(int16x4_t lowpart, int32x4_t input) {
  // CHECK: rshrn2.8h
  return vrshrn_high_n_s32(lowpart, input, 2);
}

int32x4_t test_vrshrn_high_n_s64(int32x2_t lowpart, int64x2_t input) {
  // CHECK: shrn2.4s
  return vrshrn_high_n_s64(lowpart, input, 2);
}

uint8x16_t test_vrshrn_high_n_u16(uint8x8_t lowpart, uint16x8_t input) {
  // CHECK: rshrn2.16b
  return vrshrn_high_n_u16(lowpart, input, 2);
}

uint16x8_t test_vrshrn_high_n_u32(uint16x4_t lowpart, uint32x4_t input) {
  // CHECK: rshrn2.8h
  return vrshrn_high_n_u32(lowpart, input, 2);
}

uint32x4_t test_vrshrn_high_n_u64(uint32x2_t lowpart, uint64x2_t input) {
  // CHECK: rshrn2.4s
  return vrshrn_high_n_u64(lowpart, input, 2);
}

int8x16_t test_vshrn_high_n_s16(int8x8_t lowpart, int16x8_t input) {
  // CHECK: shrn2.16b
  return vshrn_high_n_s16(lowpart, input, 2);
}

int16x8_t test_vshrn_high_n_s32(int16x4_t lowpart, int32x4_t input) {
  // CHECK: shrn2.8h
  return vshrn_high_n_s32(lowpart, input, 2);
}

int32x4_t test_vshrn_high_n_s64(int32x2_t lowpart, int64x2_t input) {
  // CHECK: shrn2.4s
  return vshrn_high_n_s64(lowpart, input, 2);
}

uint8x16_t test_vshrn_high_n_u16(uint8x8_t lowpart, uint16x8_t input) {
  // CHECK: shrn2.16b
  return vshrn_high_n_u16(lowpart, input, 2);
}

uint16x8_t test_vshrn_high_n_u32(uint16x4_t lowpart, uint32x4_t input) {
  // CHECK: shrn2.8h
  return vshrn_high_n_u32(lowpart, input, 2);
}

uint32x4_t test_vshrn_high_n_u64(uint32x2_t lowpart, uint64x2_t input) {
  // CHECK: shrn2.4s
  return vshrn_high_n_u64(lowpart, input, 2);
}

uint8x16_t test_vqshrun_high_n_s16(uint8x8_t lowpart, int16x8_t input) {
  // CHECK: sqshrun2.16b
  return vqshrun_high_n_s16(lowpart, input, 2);
}

uint16x8_t test_vqshrun_high_n_s32(uint16x4_t lowpart, int32x4_t input) {
  // CHECK: sqshrun2.8h
  return vqshrun_high_n_s32(lowpart, input, 2);
}

uint32x4_t test_vqshrun_high_n_s64(uint32x2_t lowpart, int64x2_t input) {
  // CHECK: sqshrun2.4s
  return vqshrun_high_n_s64(lowpart, input, 2);
}

uint8x16_t test_vqrshrun_high_n_s16(uint8x8_t lowpart, int16x8_t input) {
  // CHECK: sqrshrun2.16b
  return vqrshrun_high_n_s16(lowpart, input, 2);
}

uint16x8_t test_vqrshrun_high_n_s32(uint16x4_t lowpart, int32x4_t input) {
  // CHECK: sqrshrun2.8h
  return vqrshrun_high_n_s32(lowpart, input, 2);
}

uint32x4_t test_vqrshrun_high_n_s64(uint32x2_t lowpart, int64x2_t input) {
  // CHECK: sqrshrun2.4s
  return vqrshrun_high_n_s64(lowpart, input, 2);
}

int8x16_t test_vqshrn_high_n_s16(int8x8_t lowpart, int16x8_t input) {
  // CHECK: sqshrn2.16b
  return vqshrn_high_n_s16(lowpart, input, 2);
}

int16x8_t test_vqshrn_high_n_s32(int16x4_t lowpart, int32x4_t input) {
  // CHECK: sqshrn2.8h
  return vqshrn_high_n_s32(lowpart, input, 2);
}

int32x4_t test_vqshrn_high_n_s64(int32x2_t lowpart, int64x2_t input) {
  // CHECK: sqshrn2.4s
  return vqshrn_high_n_s64(lowpart, input, 2);
}

uint8x16_t test_vqshrn_high_n_u16(uint8x8_t lowpart, uint16x8_t input) {
  // CHECK: uqshrn2.16b
  return vqshrn_high_n_u16(lowpart, input, 2);
}

uint16x8_t test_vqshrn_high_n_u32(uint16x4_t lowpart, uint32x4_t input) {
  // CHECK: uqshrn2.8h
  return vqshrn_high_n_u32(lowpart, input, 2);
}

uint32x4_t test_vqshrn_high_n_u64(uint32x2_t lowpart, uint64x2_t input) {
  // CHECK: uqshrn2.4s
  return vqshrn_high_n_u64(lowpart, input, 2);
}

int8x16_t test_vqrshrn_high_n_s16(int8x8_t lowpart, int16x8_t input) {
  // CHECK: sqrshrn2.16b
  return vqrshrn_high_n_s16(lowpart, input, 2);
}

int16x8_t test_vqrshrn_high_n_s32(int16x4_t lowpart, int32x4_t input) {
  // CHECK: sqrshrn2.8h
  return vqrshrn_high_n_s32(lowpart, input, 2);
}

int32x4_t test_vqrshrn_high_n_s64(int32x2_t lowpart, int64x2_t input) {
  // CHECK: sqrshrn2.4s
  return vqrshrn_high_n_s64(lowpart, input, 2);
}

uint8x16_t test_vqrshrn_high_n_u16(uint8x8_t lowpart, uint16x8_t input) {
  // CHECK: uqrshrn2.16b
  return vqrshrn_high_n_u16(lowpart, input, 2);
}

uint16x8_t test_vqrshrn_high_n_u32(uint16x4_t lowpart, uint32x4_t input) {
  // CHECK: uqrshrn2.8h
  return vqrshrn_high_n_u32(lowpart, input, 2);
}

uint32x4_t test_vqrshrn_high_n_u64(uint32x2_t lowpart, uint64x2_t input) {
  // CHECK: uqrshrn2.4s
  return vqrshrn_high_n_u64(lowpart, input, 2);
}

int8x16_t test_vaddhn_high_s16(int8x8_t lowpart, int16x8_t lhs, int16x8_t rhs) {
  // CHECK: addhn2.16b v0, v1, v2
  return vaddhn_high_s16(lowpart, lhs, rhs);
}

int16x8_t test_vaddhn_high_s32(int16x4_t lowpart, int32x4_t lhs, int32x4_t rhs) {
  // CHECK: addhn2.8h v0, v1, v2
  return vaddhn_high_s32(lowpart, lhs, rhs);
}

int32x4_t test_vaddhn_high_s64(int32x2_t lowpart, int64x2_t lhs, int64x2_t rhs) {
  // CHECK: addhn2.4s v0, v1, v2
  return vaddhn_high_s64(lowpart, lhs, rhs);
}

uint8x16_t test_vaddhn_high_u16(uint8x8_t lowpart, uint16x8_t lhs, uint16x8_t rhs) {
  // CHECK: addhn2.16b v0, v1, v2
  return vaddhn_high_s16(lowpart, lhs, rhs);
}

uint16x8_t test_vaddhn_high_u32(uint16x4_t lowpart, uint32x4_t lhs, uint32x4_t rhs) {
  // CHECK: addhn2.8h v0, v1, v2
  return vaddhn_high_s32(lowpart, lhs, rhs);
}

uint32x4_t test_vaddhn_high_u64(uint32x2_t lowpart, uint64x2_t lhs, uint64x2_t rhs) {
  // CHECK: addhn2.4s v0, v1, v2
  return vaddhn_high_s64(lowpart, lhs, rhs);
}

int8x16_t test_vraddhn_high_s16(int8x8_t lowpart, int16x8_t lhs, int16x8_t rhs) {
  // CHECK: raddhn2.16b v0, v1, v2
  return vraddhn_high_s16(lowpart, lhs, rhs);
}

int16x8_t test_vraddhn_high_s32(int16x4_t lowpart, int32x4_t lhs, int32x4_t rhs) {
  // CHECK: raddhn2.8h v0, v1, v2
  return vraddhn_high_s32(lowpart, lhs, rhs);
}

int32x4_t test_vraddhn_high_s64(int32x2_t lowpart, int64x2_t lhs, int64x2_t rhs) {
  // CHECK: raddhn2.4s v0, v1, v2
  return vraddhn_high_s64(lowpart, lhs, rhs);
}

uint8x16_t test_vraddhn_high_u16(uint8x8_t lowpart, uint16x8_t lhs, uint16x8_t rhs) {
  // CHECK: raddhn2.16b v0, v1, v2
  return vraddhn_high_s16(lowpart, lhs, rhs);
}

uint16x8_t test_vraddhn_high_u32(uint16x4_t lowpart, uint32x4_t lhs, uint32x4_t rhs) {
  // CHECK: raddhn2.8h v0, v1, v2
  return vraddhn_high_s32(lowpart, lhs, rhs);
}

uint32x4_t test_vraddhn_high_u64(uint32x2_t lowpart, uint64x2_t lhs, uint64x2_t rhs) {
  // CHECK: raddhn2.4s v0, v1, v2
  return vraddhn_high_s64(lowpart, lhs, rhs);
}

int8x16_t test_vmovn_high_s16(int8x8_t lowpart, int16x8_t wide) {
  // CHECK: xtn2.16b v0, v1
  return vmovn_high_s16(lowpart, wide);
}

int16x8_t test_vmovn_high_s32(int16x4_t lowpart, int32x4_t wide) {
  // CHECK: xtn2.8h v0, v1
  return vmovn_high_s32(lowpart, wide);
}

int32x4_t test_vmovn_high_s64(int32x2_t lowpart, int64x2_t wide) {
  // CHECK: xtn2.4s v0, v1
  return vmovn_high_s64(lowpart, wide);
}

uint8x16_t test_vmovn_high_u16(uint8x8_t lowpart, uint16x8_t wide) {
  // CHECK: xtn2.16b v0, v1
  return vmovn_high_u16(lowpart, wide);
}

uint16x8_t test_vmovn_high_u32(uint16x4_t lowpart, uint32x4_t wide) {
  // CHECK: xtn2.8h v0, v1
  return vmovn_high_u32(lowpart, wide);
}

uint32x4_t test_vmovn_high_u64(uint32x2_t lowpart, uint64x2_t wide) {
  // CHECK: xtn2.4s v0, v1
  return vmovn_high_u64(lowpart, wide);
}

int8x16_t test_vqmovn_high_s16(int8x8_t lowpart, int16x8_t wide) {
  // CHECK: sqxtn2.16b v0, v1
  return vqmovn_high_s16(lowpart, wide);
}

int16x8_t test_vqmovn_high_s32(int16x4_t lowpart, int32x4_t wide) {
  // CHECK: sqxtn2.8h v0, v1
  return vqmovn_high_s32(lowpart, wide);
}

int32x4_t test_vqmovn_high_s64(int32x2_t lowpart, int64x2_t wide) {
  // CHECK: sqxtn2.4s v0, v1
  return vqmovn_high_s64(lowpart, wide);
}

uint8x16_t test_vqmovn_high_u16(uint8x8_t lowpart, int16x8_t wide) {
  // CHECK: uqxtn2.16b v0, v1
  return vqmovn_high_u16(lowpart, wide);
}

uint16x8_t test_vqmovn_high_u32(uint16x4_t lowpart, int32x4_t wide) {
  // CHECK: uqxtn2.8h v0, v1
  return vqmovn_high_u32(lowpart, wide);
}

uint32x4_t test_vqmovn_high_u64(uint32x2_t lowpart, int64x2_t wide) {
  // CHECK: uqxtn2.4s v0, v1
  return vqmovn_high_u64(lowpart, wide);
}

uint8x16_t test_vqmovun_high_s16(uint8x8_t lowpart, int16x8_t wide) {
  // CHECK: sqxtun2.16b v0, v1
  return vqmovun_high_s16(lowpart, wide);
}

uint16x8_t test_vqmovun_high_s32(uint16x4_t lowpart, int32x4_t wide) {
  // CHECK: sqxtun2.8h v0, v1
  return vqmovun_high_s32(lowpart, wide);
}

uint32x4_t test_vqmovun_high_s64(uint32x2_t lowpart, int64x2_t wide) {
  // CHECK: sqxtun2.4s v0, v1
  return vqmovun_high_s64(lowpart, wide);
}

float32x4_t test_vcvtx_high_f32_f64(float32x2_t lowpart, float64x2_t wide) {
  // CHECK: fcvtxn2 v0.4s, v1.2d
  return vcvtx_high_f32_f64(lowpart, wide);
}

float64x2_t test_vcvt_f64_f32(float32x2_t x) {
  // CHECK: fcvtl v0.2d, v0.2s
  return vcvt_f64_f32(x);
}

float64x2_t test_vcvt_high_f64_f32(float32x4_t x) {
  // CHECK: fcvtl2 v0.2d, v0.4s
  return vcvt_high_f64_f32(x);
}

float32x2_t test_vcvt_f32_f64(float64x2_t v) {
  // CHECK: fcvtn v0.2s, v0.2d
  return vcvt_f32_f64(v);
}

float32x4_t test_vcvt_high_f32_f64(float32x2_t x, float64x2_t v) {
  // CHECK: fcvtn2 v0.4s, v1.2d
  return vcvt_high_f32_f64(x, v);
}

float32x2_t test_vcvtx_f32_f64(float64x2_t v) {
  // CHECK: fcvtxn v0.2s, v0.2d
  return vcvtx_f32_f64(v);
}
