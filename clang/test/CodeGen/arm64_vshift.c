// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -ffreestanding -emit-llvm -o - -O1 %s | FileCheck %s
#include <arm_neon.h>

int8x8_t test_vqshl_n_s8(int8x8_t in) {
  // CHECK-LABEL: @test_vqshl_n_s8
  // CHECK: call <8 x i8> @llvm.arm64.neon.sqshl.v8i8(<8 x i8> %in, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshl_n_s8(in, 1);
}

int16x4_t test_vqshl_n_s16(int16x4_t in) {
  // CHECK-LABEL: @test_vqshl_n_s16
  // CHECK: call <4 x i16> @llvm.arm64.neon.sqshl.v4i16(<4 x i16> %in, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
  return vqshl_n_s16(in, 1);
}

int32x2_t test_vqshl_n_s32(int32x2_t in) {
  // CHECK-LABEL: @test_vqshl_n_s32
  // CHECK: call <2 x i32> @llvm.arm64.neon.sqshl.v2i32(<2 x i32> %in, <2 x i32> <i32 1, i32 1>)
  return vqshl_n_s32(in, 1);
}

int64x1_t test_vqshl_n_s64(int64x1_t in) {
  // CHECK-LABEL: @test_vqshl_n_s64
  // CHECK: call <1 x i64> @llvm.arm64.neon.sqshl.v1i64(<1 x i64> %in, <1 x i64> <i64 1>)
  return vqshl_n_s64(in, 1);
}


int8x16_t test_vqshlq_n_s8(int8x16_t in) {
  // CHECK-LABEL: @test_vqshlq_n_s8
  // CHECK: call <16 x i8> @llvm.arm64.neon.sqshl.v16i8(<16 x i8> %in, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshlq_n_s8(in, 1);
}

int16x8_t test_vqshlq_n_s16(int16x8_t in) {
  // CHECK-LABEL: @test_vqshlq_n_s16
  // CHECK: call <8 x i16> @llvm.arm64.neon.sqshl.v8i16(<8 x i16> %in, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  return vqshlq_n_s16(in, 1);
}

int32x4_t test_vqshlq_n_s32(int32x4_t in) {
  // CHECK-LABEL: @test_vqshlq_n_s32
  // CHECK: call <4 x i32> @llvm.arm64.neon.sqshl.v4i32(<4 x i32> %in, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  return vqshlq_n_s32(in, 1);
}

int64x2_t test_vqshlq_n_s64(int64x2_t in) {
  // CHECK-LABEL: @test_vqshlq_n_s64
  // CHECK: call <2 x i64> @llvm.arm64.neon.sqshl.v2i64(<2 x i64> %in, <2 x i64> <i64 1, i64 1>
  return vqshlq_n_s64(in, 1);
}

uint8x8_t test_vqshl_n_u8(uint8x8_t in) {
  // CHECK-LABEL: @test_vqshl_n_u8
  // CHECK: call <8 x i8> @llvm.arm64.neon.uqshl.v8i8(<8 x i8> %in, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshl_n_u8(in, 1);
}

uint16x4_t test_vqshl_n_u16(uint16x4_t in) {
  // CHECK-LABEL: @test_vqshl_n_u16
  // CHECK: call <4 x i16> @llvm.arm64.neon.uqshl.v4i16(<4 x i16> %in, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
  return vqshl_n_u16(in, 1);
}

uint32x2_t test_vqshl_n_u32(uint32x2_t in) {
  // CHECK-LABEL: @test_vqshl_n_u32
  // CHECK: call <2 x i32> @llvm.arm64.neon.uqshl.v2i32(<2 x i32> %in, <2 x i32> <i32 1, i32 1>)
  return vqshl_n_u32(in, 1);
}

uint64x1_t test_vqshl_n_u64(uint64x1_t in) {
  // CHECK-LABEL: @test_vqshl_n_u64
  // CHECK: call <1 x i64> @llvm.arm64.neon.uqshl.v1i64(<1 x i64> %in, <1 x i64> <i64 1>)
  return vqshl_n_u64(in, 1);
}

uint8x16_t test_vqshlq_n_u8(uint8x16_t in) {
  // CHECK-LABEL: @test_vqshlq_n_u8
  // CHECK: call <16 x i8> @llvm.arm64.neon.uqshl.v16i8(<16 x i8> %in, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshlq_n_u8(in, 1);
}

uint16x8_t test_vqshlq_n_u16(uint16x8_t in) {
  // CHECK-LABEL: @test_vqshlq_n_u16
  // CHECK: call <8 x i16> @llvm.arm64.neon.uqshl.v8i16(<8 x i16> %in, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  return vqshlq_n_u16(in, 1);
}

uint32x4_t test_vqshlq_n_u32(uint32x4_t in) {
  // CHECK-LABEL: @test_vqshlq_n_u32
  // CHECK: call <4 x i32> @llvm.arm64.neon.uqshl.v4i32(<4 x i32> %in, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  return vqshlq_n_u32(in, 1);
}

uint64x2_t test_vqshlq_n_u64(uint64x2_t in) {
  // CHECK-LABEL: @test_vqshlq_n_u64
  // CHECK: call <2 x i64> @llvm.arm64.neon.uqshl.v2i64(<2 x i64> %in, <2 x i64> <i64 1, i64 1>
  return vqshlq_n_u64(in, 1);
}

int8x8_t test_vrshr_n_s8(int8x8_t in) {
  // CHECK-LABEL: @test_vrshr_n_s8
  // CHECK: call <8 x i8> @llvm.arm64.neon.srshl.v8i8(<8 x i8> %in, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  return vrshr_n_s8(in, 1);
}

int16x4_t test_vrshr_n_s16(int16x4_t in) {
  // CHECK-LABEL: @test_vrshr_n_s16
  // CHECK: call <4 x i16> @llvm.arm64.neon.srshl.v4i16(<4 x i16> %in, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
  return vrshr_n_s16(in, 1);
}

int32x2_t test_vrshr_n_s32(int32x2_t in) {
  // CHECK-LABEL: @test_vrshr_n_s32
  // CHECK: call <2 x i32> @llvm.arm64.neon.srshl.v2i32(<2 x i32> %in, <2 x i32> <i32 -1, i32 -1>)
  return vrshr_n_s32(in, 1);
}

int64x1_t test_vrshr_n_s64(int64x1_t in) {
  // CHECK-LABEL: @test_vrshr_n_s64
  // CHECK: call <1 x i64> @llvm.arm64.neon.srshl.v1i64(<1 x i64> %in, <1 x i64> <i64 -1>)
  return vrshr_n_s64(in, 1);
}


int8x16_t test_vrshrq_n_s8(int8x16_t in) {
  // CHECK-LABEL: @test_vrshrq_n_s8
  // CHECK: call <16 x i8> @llvm.arm64.neon.srshl.v16i8(<16 x i8> %in, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  return vrshrq_n_s8(in, 1);
}

int16x8_t test_vrshrq_n_s16(int16x8_t in) {
  // CHECK-LABEL: @test_vrshrq_n_s16
  // CHECK: call <8 x i16> @llvm.arm64.neon.srshl.v8i16(<8 x i16> %in, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
  return vrshrq_n_s16(in, 1);
}

int32x4_t test_vrshrq_n_s32(int32x4_t in) {
  // CHECK-LABEL: @test_vrshrq_n_s32
  // CHECK: call <4 x i32> @llvm.arm64.neon.srshl.v4i32(<4 x i32> %in, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
  return vrshrq_n_s32(in, 1);
}

int64x2_t test_vrshrq_n_s64(int64x2_t in) {
  // CHECK-LABEL: @test_vrshrq_n_s64
  // CHECK: call <2 x i64> @llvm.arm64.neon.srshl.v2i64(<2 x i64> %in, <2 x i64> <i64 -1, i64 -1>
  return vrshrq_n_s64(in, 1);
}

uint8x8_t test_vrshr_n_u8(uint8x8_t in) {
  // CHECK-LABEL: @test_vrshr_n_u8
  // CHECK: call <8 x i8> @llvm.arm64.neon.urshl.v8i8(<8 x i8> %in, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  return vrshr_n_u8(in, 1);
}

uint16x4_t test_vrshr_n_u16(uint16x4_t in) {
  // CHECK-LABEL: @test_vrshr_n_u16
  // CHECK: call <4 x i16> @llvm.arm64.neon.urshl.v4i16(<4 x i16> %in, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
  return vrshr_n_u16(in, 1);
}

uint32x2_t test_vrshr_n_u32(uint32x2_t in) {
  // CHECK-LABEL: @test_vrshr_n_u32
  // CHECK: call <2 x i32> @llvm.arm64.neon.urshl.v2i32(<2 x i32> %in, <2 x i32> <i32 -1, i32 -1>)
  return vrshr_n_u32(in, 1);
}

uint64x1_t test_vrshr_n_u64(uint64x1_t in) {
  // CHECK-LABEL: @test_vrshr_n_u64
  // CHECK: call <1 x i64> @llvm.arm64.neon.urshl.v1i64(<1 x i64> %in, <1 x i64> <i64 -1>)
  return vrshr_n_u64(in, 1);
}

uint8x16_t test_vrshrq_n_u8(uint8x16_t in) {
  // CHECK-LABEL: @test_vrshrq_n_u8
  // CHECK: call <16 x i8> @llvm.arm64.neon.urshl.v16i8(<16 x i8> %in, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  return vrshrq_n_u8(in, 1);
}

uint16x8_t test_vrshrq_n_u16(uint16x8_t in) {
  // CHECK-LABEL: @test_vrshrq_n_u16
  // CHECK: call <8 x i16> @llvm.arm64.neon.urshl.v8i16(<8 x i16> %in, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
  return vrshrq_n_u16(in, 1);
}

uint32x4_t test_vrshrq_n_u32(uint32x4_t in) {
  // CHECK-LABEL: @test_vrshrq_n_u32
  // CHECK: call <4 x i32> @llvm.arm64.neon.urshl.v4i32(<4 x i32> %in, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
  return vrshrq_n_u32(in, 1);
}

uint64x2_t test_vrshrq_n_u64(uint64x2_t in) {
  // CHECK-LABEL: @test_vrshrq_n_u64
  // CHECK: call <2 x i64> @llvm.arm64.neon.urshl.v2i64(<2 x i64> %in, <2 x i64> <i64 -1, i64 -1>
  return vrshrq_n_u64(in, 1);
}

int8x8_t test_vqshlu_n_s8(int8x8_t in) {
  // CHECK-LABEL: @test_vqshlu_n_s8
  // CHECK: call <8 x i8> @llvm.arm64.neon.sqshlu.v8i8(<8 x i8> %in, <8 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshlu_n_s8(in, 1);
}

int16x4_t test_vqshlu_n_s16(int16x4_t in) {
  // CHECK-LABEL: @test_vqshlu_n_s16
  // CHECK: call <4 x i16> @llvm.arm64.neon.sqshlu.v4i16(<4 x i16> %in, <4 x i16> <i16 1, i16 1, i16 1, i16 1>)
  return vqshlu_n_s16(in, 1);
}

int32x2_t test_vqshlu_n_s32(int32x2_t in) {
  // CHECK-LABEL: @test_vqshlu_n_s32
  // CHECK: call <2 x i32> @llvm.arm64.neon.sqshlu.v2i32(<2 x i32> %in, <2 x i32> <i32 1, i32 1>)
  return vqshlu_n_s32(in, 1);
}

int64x1_t test_vqshlu_n_s64(int64x1_t in) {
  // CHECK-LABEL: @test_vqshlu_n_s64
  // CHECK: call <1 x i64> @llvm.arm64.neon.sqshlu.v1i64(<1 x i64> %in, <1 x i64> <i64 1>)
  return vqshlu_n_s64(in, 1);
}


int8x16_t test_vqshluq_n_s8(int8x16_t in) {
  // CHECK-LABEL: @test_vqshluq_n_s8
  // CHECK: call <16 x i8> @llvm.arm64.neon.sqshlu.v16i8(<16 x i8> %in, <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  return vqshluq_n_s8(in, 1);
}

int16x8_t test_vqshluq_n_s16(int16x8_t in) {
  // CHECK-LABEL: @test_vqshluq_n_s16
  // CHECK: call <8 x i16> @llvm.arm64.neon.sqshlu.v8i16(<8 x i16> %in, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  return vqshluq_n_s16(in, 1);
}

int32x4_t test_vqshluq_n_s32(int32x4_t in) {
  // CHECK-LABEL: @test_vqshluq_n_s32
  // CHECK: call <4 x i32> @llvm.arm64.neon.sqshlu.v4i32(<4 x i32> %in, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  return vqshluq_n_s32(in, 1);
}

int64x2_t test_vqshluq_n_s64(int64x2_t in) {
  // CHECK-LABEL: @test_vqshluq_n_s64
  // CHECK: call <2 x i64> @llvm.arm64.neon.sqshlu.v2i64(<2 x i64> %in, <2 x i64> <i64 1, i64 1>
  return vqshluq_n_s64(in, 1);
}

int8x8_t test_vrsra_n_s8(int8x8_t acc, int8x8_t in) {
  // CHECK-LABEL: @test_vrsra_n_s8
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <8 x i8> @llvm.arm64.neon.srshl.v8i8(<8 x i8> %in, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  // CHECK: add <8 x i8> [[TMP]], %acc
  return vrsra_n_s8(acc, in, 1);
}

int16x4_t test_vrsra_n_s16(int16x4_t acc, int16x4_t in) {
  // CHECK-LABEL: @test_vrsra_n_s16
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <4 x i16> @llvm.arm64.neon.srshl.v4i16(<4 x i16> %in, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
  // CHECK: add <4 x i16> [[TMP]], %acc
  return vrsra_n_s16(acc, in, 1);
}

int32x2_t test_vrsra_n_s32(int32x2_t acc, int32x2_t in) {
  // CHECK-LABEL: @test_vrsra_n_s32
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <2 x i32> @llvm.arm64.neon.srshl.v2i32(<2 x i32> %in, <2 x i32> <i32 -1, i32 -1>)
  // CHECK: add <2 x i32> [[TMP]], %acc
  return vrsra_n_s32(acc, in, 1);
}

int64x1_t test_vrsra_n_s64(int64x1_t acc, int64x1_t in) {
  // CHECK-LABEL: @test_vrsra_n_s64
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <1 x i64> @llvm.arm64.neon.srshl.v1i64(<1 x i64> %in, <1 x i64> <i64 -1>)
  // CHECK: add <1 x i64> [[TMP]], %acc
  return vrsra_n_s64(acc, in, 1);
}

int8x16_t test_vrsraq_n_s8(int8x16_t acc, int8x16_t in) {
  // CHECK-LABEL: @test_vrsraq_n_s8
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <16 x i8> @llvm.arm64.neon.srshl.v16i8(<16 x i8> %in, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  // CHECK: add <16 x i8> [[TMP]], %acc
  return vrsraq_n_s8(acc, in, 1);
}

int16x8_t test_vrsraq_n_s16(int16x8_t acc, int16x8_t in) {
  // CHECK-LABEL: @test_vrsraq_n_s16
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <8 x i16> @llvm.arm64.neon.srshl.v8i16(<8 x i16> %in, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
  // CHECK: add <8 x i16> [[TMP]], %acc
  return vrsraq_n_s16(acc, in, 1);
}

int32x4_t test_vrsraq_n_s32(int32x4_t acc, int32x4_t in) {
  // CHECK-LABEL: @test_vrsraq_n_s32
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <4 x i32> @llvm.arm64.neon.srshl.v4i32(<4 x i32> %in, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
  // CHECK: add <4 x i32> [[TMP]], %acc
  return vrsraq_n_s32(acc, in, 1);
}

int64x2_t test_vrsraq_n_s64(int64x2_t acc, int64x2_t in) {
  // CHECK-LABEL: @test_vrsraq_n_s64
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <2 x i64> @llvm.arm64.neon.srshl.v2i64(<2 x i64> %in, <2 x i64> <i64 -1, i64 -1>)
  // CHECK: add <2 x i64> [[TMP]], %acc
  return vrsraq_n_s64(acc, in, 1);
}

uint8x8_t test_vrsra_n_u8(uint8x8_t acc, uint8x8_t in) {
  // CHECK-LABEL: @test_vrsra_n_u8
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <8 x i8> @llvm.arm64.neon.urshl.v8i8(<8 x i8> %in, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  // CHECK: add <8 x i8> [[TMP]], %acc
  return vrsra_n_u8(acc, in, 1);
}

uint16x4_t test_vrsra_n_u16(uint16x4_t acc, uint16x4_t in) {
  // CHECK-LABEL: @test_vrsra_n_u16
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <4 x i16> @llvm.arm64.neon.urshl.v4i16(<4 x i16> %in, <4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1>)
  // CHECK: add <4 x i16> [[TMP]], %acc
  return vrsra_n_u16(acc, in, 1);
}

uint32x2_t test_vrsra_n_u32(uint32x2_t acc, uint32x2_t in) {
  // CHECK-LABEL: @test_vrsra_n_u32
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <2 x i32> @llvm.arm64.neon.urshl.v2i32(<2 x i32> %in, <2 x i32> <i32 -1, i32 -1>)
  // CHECK: add <2 x i32> [[TMP]], %acc
  return vrsra_n_u32(acc, in, 1);
}

uint64x1_t test_vrsra_n_u64(uint64x1_t acc, uint64x1_t in) {
  // CHECK-LABEL: @test_vrsra_n_u64
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <1 x i64> @llvm.arm64.neon.urshl.v1i64(<1 x i64> %in, <1 x i64> <i64 -1>)
  // CHECK: add <1 x i64> [[TMP]], %acc
  return vrsra_n_u64(acc, in, 1);
}

uint8x16_t test_vrsraq_n_u8(uint8x16_t acc, uint8x16_t in) {
  // CHECK-LABEL: @test_vrsraq_n_u8
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <16 x i8> @llvm.arm64.neon.urshl.v16i8(<16 x i8> %in, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>)
  // CHECK: add <16 x i8> [[TMP]], %acc
  return vrsraq_n_u8(acc, in, 1);
}

uint16x8_t test_vrsraq_n_u16(uint16x8_t acc, uint16x8_t in) {
  // CHECK-LABEL: @test_vrsraq_n_u16
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <8 x i16> @llvm.arm64.neon.urshl.v8i16(<8 x i16> %in, <8 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>)
  // CHECK: add <8 x i16> [[TMP]], %acc
  return vrsraq_n_u16(acc, in, 1);
}

uint32x4_t test_vrsraq_n_u32(uint32x4_t acc, uint32x4_t in) {
  // CHECK-LABEL: @test_vrsraq_n_u32
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <4 x i32> @llvm.arm64.neon.urshl.v4i32(<4 x i32> %in, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>)
  // CHECK: add <4 x i32> [[TMP]], %acc
  return vrsraq_n_u32(acc, in, 1);
}

uint64x2_t test_vrsraq_n_u64(uint64x2_t acc, uint64x2_t in) {
  // CHECK-LABEL: @test_vrsraq_n_u64
  // CHECK: [[TMP:%[0-9a-zA-Z._]+]] = tail call <2 x i64> @llvm.arm64.neon.urshl.v2i64(<2 x i64> %in, <2 x i64> <i64 -1, i64 -1>)
  // CHECK: add <2 x i64> [[TMP]], %acc
  return vrsraq_n_u64(acc, in, 1);
}
