; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

%struct.int8x16x2_t = type { [2 x <16 x i8>] }
%struct.int16x8x2_t = type { [2 x <8 x i16>] }
%struct.int32x4x2_t = type { [2 x <4 x i32>] }
%struct.int64x2x2_t = type { [2 x <2 x i64>] }
%struct.float32x4x2_t = type { [2 x <4 x float>] }
%struct.float64x2x2_t = type { [2 x <2 x double>] }
%struct.int8x8x2_t = type { [2 x <8 x i8>] }
%struct.int16x4x2_t = type { [2 x <4 x i16>] }
%struct.int32x2x2_t = type { [2 x <2 x i32>] }
%struct.int64x1x2_t = type { [2 x <1 x i64>] }
%struct.float32x2x2_t = type { [2 x <2 x float>] }
%struct.float64x1x2_t = type { [2 x <1 x double>] }
%struct.int8x16x3_t = type { [3 x <16 x i8>] }
%struct.int16x8x3_t = type { [3 x <8 x i16>] }
%struct.int32x4x3_t = type { [3 x <4 x i32>] }
%struct.int64x2x3_t = type { [3 x <2 x i64>] }
%struct.float32x4x3_t = type { [3 x <4 x float>] }
%struct.float64x2x3_t = type { [3 x <2 x double>] }
%struct.int8x8x3_t = type { [3 x <8 x i8>] }
%struct.int16x4x3_t = type { [3 x <4 x i16>] }
%struct.int32x2x3_t = type { [3 x <2 x i32>] }
%struct.int64x1x3_t = type { [3 x <1 x i64>] }
%struct.float32x2x3_t = type { [3 x <2 x float>] }
%struct.float64x1x3_t = type { [3 x <1 x double>] }
%struct.int8x16x4_t = type { [4 x <16 x i8>] }
%struct.int16x8x4_t = type { [4 x <8 x i16>] }
%struct.int32x4x4_t = type { [4 x <4 x i32>] }
%struct.int64x2x4_t = type { [4 x <2 x i64>] }
%struct.float32x4x4_t = type { [4 x <4 x float>] }
%struct.float64x2x4_t = type { [4 x <2 x double>] }
%struct.int8x8x4_t = type { [4 x <8 x i8>] }
%struct.int16x4x4_t = type { [4 x <4 x i16>] }
%struct.int32x2x4_t = type { [4 x <2 x i32>] }
%struct.int64x1x4_t = type { [4 x <1 x i64>] }
%struct.float32x2x4_t = type { [4 x <2 x float>] }
%struct.float64x1x4_t = type { [4 x <1 x double>] }


define <16 x i8> @test_vld1q_s8(i8* readonly %a) {
; CHECK: test_vld1q_s8
; CHECK: ld1 {v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %a, i32 1)
  ret <16 x i8> %vld1
}

define <8 x i16> @test_vld1q_s16(i16* readonly %a) {
; CHECK: test_vld1q_s16
; CHECK: ld1 {v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %1, i32 2)
  ret <8 x i16> %vld1
}

define <4 x i32> @test_vld1q_s32(i32* readonly %a) {
; CHECK: test_vld1q_s32
; CHECK: ld1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld1 = tail call <4 x i32> @llvm.arm.neon.vld1.v4i32(i8* %1, i32 4)
  ret <4 x i32> %vld1
}

define <2 x i64> @test_vld1q_s64(i64* readonly %a) {
; CHECK: test_vld1q_s64
; CHECK: ld1 {v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld1 = tail call <2 x i64> @llvm.arm.neon.vld1.v2i64(i8* %1, i32 8)
  ret <2 x i64> %vld1
}

define <4 x float> @test_vld1q_f32(float* readonly %a) {
; CHECK: test_vld1q_f32
; CHECK: ld1 {v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %1, i32 4)
  ret <4 x float> %vld1
}

define <2 x double> @test_vld1q_f64(double* readonly %a) {
; CHECK: test_vld1q_f64
; CHECK: ld1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld1 = tail call <2 x double> @llvm.arm.neon.vld1.v2f64(i8* %1, i32 8)
  ret <2 x double> %vld1
}

define <8 x i8> @test_vld1_s8(i8* readonly %a) {
; CHECK: test_vld1_s8
; CHECK: ld1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %a, i32 1)
  ret <8 x i8> %vld1
}

define <4 x i16> @test_vld1_s16(i16* readonly %a) {
; CHECK: test_vld1_s16
; CHECK: ld1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %1, i32 2)
  ret <4 x i16> %vld1
}

define <2 x i32> @test_vld1_s32(i32* readonly %a) {
; CHECK: test_vld1_s32
; CHECK: ld1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld1 = tail call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %1, i32 4)
  ret <2 x i32> %vld1
}

define <1 x i64> @test_vld1_s64(i64* readonly %a) {
; CHECK: test_vld1_s64
; CHECK: ld1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld1 = tail call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %1, i32 8)
  ret <1 x i64> %vld1
}

define <2 x float> @test_vld1_f32(float* readonly %a) {
; CHECK: test_vld1_f32
; CHECK: ld1 {v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld1 = tail call <2 x float> @llvm.arm.neon.vld1.v2f32(i8* %1, i32 4)
  ret <2 x float> %vld1
}

define <1 x double> @test_vld1_f64(double* readonly %a) {
; CHECK: test_vld1_f64
; CHECK: ld1 {v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld1 = tail call <1 x double> @llvm.arm.neon.vld1.v1f64(i8* %1, i32 8)
  ret <1 x double> %vld1
}

define <8 x i8> @test_vld1_p8(i8* readonly %a) {
; CHECK: test_vld1_p8
; CHECK: ld1 {v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld1 = tail call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %a, i32 1)
  ret <8 x i8> %vld1
}

define <4 x i16> @test_vld1_p16(i16* readonly %a) {
; CHECK: test_vld1_p16
; CHECK: ld1 {v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld1 = tail call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %1, i32 2)
  ret <4 x i16> %vld1
}

define %struct.int8x16x2_t @test_vld2q_s8(i8* readonly %a) {
; CHECK: test_vld2q_s8
; CHECK: ld2 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld2 = tail call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8* %a, i32 1)
  %vld2.fca.0.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <16 x i8>, <16 x i8> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %vld2.fca.1.extract, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vld2q_s16(i16* readonly %a) {
; CHECK: test_vld2q_s16
; CHECK: ld2 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld2 = tail call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2.v8i16(i8* %1, i32 2)
  %vld2.fca.0.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vld2.fca.1.extract, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vld2q_s32(i32* readonly %a) {
; CHECK: test_vld2q_s32
; CHECK: ld2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld2 = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vld2.fca.1.extract, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.int64x2x2_t @test_vld2q_s64(i64* readonly %a) {
; CHECK: test_vld2q_s64
; CHECK: ld2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld2 = tail call { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2.v2i64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int64x2x2_t undef, <2 x i64> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x2_t %.fca.0.0.insert, <2 x i64> %vld2.fca.1.extract, 0, 1
  ret %struct.int64x2x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vld2q_f32(float* readonly %a) {
; CHECK: test_vld2q_f32
; CHECK: ld2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <4 x float>, <4 x float> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vld2.fca.1.extract, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.float64x2x2_t @test_vld2q_f64(double* readonly %a) {
; CHECK: test_vld2q_f64
; CHECK: ld2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld2 = tail call { <2 x double>, <2 x double> } @llvm.arm.neon.vld2.v2f64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <2 x double>, <2 x double> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x double>, <2 x double> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float64x2x2_t undef, <2 x double> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x2_t %.fca.0.0.insert, <2 x double> %vld2.fca.1.extract, 0, 1
  ret %struct.float64x2x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vld2_s8(i8* readonly %a) {
; CHECK: test_vld2_s8
; CHECK: ld2 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld2 = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8* %a, i32 1)
  %vld2.fca.0.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vld2.fca.1.extract, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vld2_s16(i16* readonly %a) {
; CHECK: test_vld2_s16
; CHECK: ld2 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld2 = tail call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2.v4i16(i8* %1, i32 2)
  %vld2.fca.0.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vld2.fca.1.extract, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vld2_s32(i32* readonly %a) {
; CHECK: test_vld2_s32
; CHECK: ld2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld2 = tail call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vld2.fca.1.extract, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.int64x1x2_t @test_vld2_s64(i64* readonly %a) {
; CHECK: test_vld2_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld2 = tail call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.int64x1x2_t undef, <1 x i64> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x2_t %.fca.0.0.insert, <1 x i64> %vld2.fca.1.extract, 0, 1
  ret %struct.int64x1x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vld2_f32(float* readonly %a) {
; CHECK: test_vld2_f32
; CHECK: ld2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld2 = tail call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8* %1, i32 4)
  %vld2.fca.0.extract = extractvalue { <2 x float>, <2 x float> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <2 x float>, <2 x float> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vld2.fca.1.extract, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.float64x1x2_t @test_vld2_f64(double* readonly %a) {
; CHECK: test_vld2_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld2 = tail call { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8* %1, i32 8)
  %vld2.fca.0.extract = extractvalue { <1 x double>, <1 x double> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <1 x double>, <1 x double> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.float64x1x2_t undef, <1 x double> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x2_t %.fca.0.0.insert, <1 x double> %vld2.fca.1.extract, 0, 1
  ret %struct.float64x1x2_t %.fca.0.1.insert
}

define %struct.int8x16x3_t @test_vld3q_s8(i8* readonly %a) {
; CHECK: test_vld3q_s8
; CHECK: ld3 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld3 = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8* %a, i32 1)
  %vld3.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int8x16x3_t undef, <16 x i8> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x3_t %.fca.0.0.insert, <16 x i8> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x3_t %.fca.0.1.insert, <16 x i8> %vld3.fca.2.extract, 0, 2
  ret %struct.int8x16x3_t %.fca.0.2.insert
}

define %struct.int16x8x3_t @test_vld3q_s16(i16* readonly %a) {
; CHECK: test_vld3q_s16
; CHECK: ld3 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld3 = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3.v8i16(i8* %1, i32 2)
  %vld3.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int16x8x3_t undef, <8 x i16> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x3_t %.fca.0.0.insert, <8 x i16> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x3_t %.fca.0.1.insert, <8 x i16> %vld3.fca.2.extract, 0, 2
  ret %struct.int16x8x3_t %.fca.0.2.insert
}

define %struct.int32x4x3_t @test_vld3q_s32(i32* readonly %a) {
; CHECK: test_vld3q_s32
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld3 = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int32x4x3_t undef, <4 x i32> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x3_t %.fca.0.0.insert, <4 x i32> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x3_t %.fca.0.1.insert, <4 x i32> %vld3.fca.2.extract, 0, 2
  ret %struct.int32x4x3_t %.fca.0.2.insert
}

define %struct.int64x2x3_t @test_vld3q_s64(i64* readonly %a) {
; CHECK: test_vld3q_s64
; CHECK: ld3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld3 = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3.v2i64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int64x2x3_t undef, <2 x i64> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x3_t %.fca.0.0.insert, <2 x i64> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x3_t %.fca.0.1.insert, <2 x i64> %vld3.fca.2.extract, 0, 2
  ret %struct.int64x2x3_t %.fca.0.2.insert
}

define %struct.float32x4x3_t @test_vld3q_f32(float* readonly %a) {
; CHECK: test_vld3q_f32
; CHECK: ld3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld3 = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float32x4x3_t undef, <4 x float> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x3_t %.fca.0.0.insert, <4 x float> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x3_t %.fca.0.1.insert, <4 x float> %vld3.fca.2.extract, 0, 2
  ret %struct.float32x4x3_t %.fca.0.2.insert
}

define %struct.float64x2x3_t @test_vld3q_f64(double* readonly %a) {
; CHECK: test_vld3q_f64
; CHECK: ld3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld3 = tail call { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3.v2f64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float64x2x3_t undef, <2 x double> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x3_t %.fca.0.0.insert, <2 x double> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x3_t %.fca.0.1.insert, <2 x double> %vld3.fca.2.extract, 0, 2
  ret %struct.float64x2x3_t %.fca.0.2.insert
}

define %struct.int8x8x3_t @test_vld3_s8(i8* readonly %a) {
; CHECK: test_vld3_s8
; CHECK: ld3 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld3 = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8* %a, i32 1)
  %vld3.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int8x8x3_t undef, <8 x i8> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x3_t %.fca.0.0.insert, <8 x i8> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x3_t %.fca.0.1.insert, <8 x i8> %vld3.fca.2.extract, 0, 2
  ret %struct.int8x8x3_t %.fca.0.2.insert
}

define %struct.int16x4x3_t @test_vld3_s16(i16* readonly %a) {
; CHECK: test_vld3_s16
; CHECK: ld3 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld3 = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8* %1, i32 2)
  %vld3.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int16x4x3_t undef, <4 x i16> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x3_t %.fca.0.0.insert, <4 x i16> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x3_t %.fca.0.1.insert, <4 x i16> %vld3.fca.2.extract, 0, 2
  ret %struct.int16x4x3_t %.fca.0.2.insert
}

define %struct.int32x2x3_t @test_vld3_s32(i32* readonly %a) {
; CHECK: test_vld3_s32
; CHECK: ld3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld3 = tail call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int32x2x3_t undef, <2 x i32> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x3_t %.fca.0.0.insert, <2 x i32> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x3_t %.fca.0.1.insert, <2 x i32> %vld3.fca.2.extract, 0, 2
  ret %struct.int32x2x3_t %.fca.0.2.insert
}

define %struct.int64x1x3_t @test_vld3_s64(i64* readonly %a) {
; CHECK: test_vld3_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld3 = tail call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.int64x1x3_t undef, <1 x i64> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x3_t %.fca.0.0.insert, <1 x i64> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x3_t %.fca.0.1.insert, <1 x i64> %vld3.fca.2.extract, 0, 2
  ret %struct.int64x1x3_t %.fca.0.2.insert
}

define %struct.float32x2x3_t @test_vld3_f32(float* readonly %a) {
; CHECK: test_vld3_f32
; CHECK: ld3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld3 = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3.v2f32(i8* %1, i32 4)
  %vld3.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float32x2x3_t undef, <2 x float> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x3_t %.fca.0.0.insert, <2 x float> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x3_t %.fca.0.1.insert, <2 x float> %vld3.fca.2.extract, 0, 2
  ret %struct.float32x2x3_t %.fca.0.2.insert
}

define %struct.float64x1x3_t @test_vld3_f64(double* readonly %a) {
; CHECK: test_vld3_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld3 = tail call { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8* %1, i32 8)
  %vld3.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.float64x1x3_t undef, <1 x double> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x3_t %.fca.0.0.insert, <1 x double> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x3_t %.fca.0.1.insert, <1 x double> %vld3.fca.2.extract, 0, 2
  ret %struct.float64x1x3_t %.fca.0.2.insert
}

define %struct.int8x16x4_t @test_vld4q_s8(i8* readonly %a) {
; CHECK: test_vld4q_s8
; CHECK: ld4 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [x{{[0-9]+|sp}}]
  %vld4 = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8* %a, i32 1)
  %vld4.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int8x16x4_t undef, <16 x i8> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x4_t %.fca.0.0.insert, <16 x i8> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x4_t %.fca.0.1.insert, <16 x i8> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x16x4_t %.fca.0.2.insert, <16 x i8> %vld4.fca.3.extract, 0, 3
  ret %struct.int8x16x4_t %.fca.0.3.insert
}

define %struct.int16x8x4_t @test_vld4q_s16(i16* readonly %a) {
; CHECK: test_vld4q_s16
; CHECK: ld4 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld4 = tail call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8* %1, i32 2)
  %vld4.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int16x8x4_t undef, <8 x i16> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x4_t %.fca.0.0.insert, <8 x i16> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x4_t %.fca.0.1.insert, <8 x i16> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x8x4_t %.fca.0.2.insert, <8 x i16> %vld4.fca.3.extract, 0, 3
  ret %struct.int16x8x4_t %.fca.0.3.insert
}

define %struct.int32x4x4_t @test_vld4q_s32(i32* readonly %a) {
; CHECK: test_vld4q_s32
; CHECK: ld4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld4 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int32x4x4_t undef, <4 x i32> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x4_t %.fca.0.0.insert, <4 x i32> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x4_t %.fca.0.1.insert, <4 x i32> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x4x4_t %.fca.0.2.insert, <4 x i32> %vld4.fca.3.extract, 0, 3
  ret %struct.int32x4x4_t %.fca.0.3.insert
}

define %struct.int64x2x4_t @test_vld4q_s64(i64* readonly %a) {
; CHECK: test_vld4q_s64
; CHECK: ld4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld4 = tail call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4.v2i64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int64x2x4_t undef, <2 x i64> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x4_t %.fca.0.0.insert, <2 x i64> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x4_t %.fca.0.1.insert, <2 x i64> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x2x4_t %.fca.0.2.insert, <2 x i64> %vld4.fca.3.extract, 0, 3
  ret %struct.int64x2x4_t %.fca.0.3.insert
}

define %struct.float32x4x4_t @test_vld4q_f32(float* readonly %a) {
; CHECK: test_vld4q_f32
; CHECK: ld4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld4 = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float32x4x4_t undef, <4 x float> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x4_t %.fca.0.0.insert, <4 x float> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x4_t %.fca.0.1.insert, <4 x float> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x4x4_t %.fca.0.2.insert, <4 x float> %vld4.fca.3.extract, 0, 3
  ret %struct.float32x4x4_t %.fca.0.3.insert
}

define %struct.float64x2x4_t @test_vld4q_f64(double* readonly %a) {
; CHECK: test_vld4q_f64
; CHECK: ld4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld4 = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4.v2f64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float64x2x4_t undef, <2 x double> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x4_t %.fca.0.0.insert, <2 x double> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x4_t %.fca.0.1.insert, <2 x double> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x2x4_t %.fca.0.2.insert, <2 x double> %vld4.fca.3.extract, 0, 3
  ret %struct.float64x2x4_t %.fca.0.3.insert
}

define %struct.int8x8x4_t @test_vld4_s8(i8* readonly %a) {
; CHECK: test_vld4_s8
; CHECK: ld4 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [x{{[0-9]+|sp}}]
  %vld4 = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8* %a, i32 1)
  %vld4.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int8x8x4_t undef, <8 x i8> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x4_t %.fca.0.0.insert, <8 x i8> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x4_t %.fca.0.1.insert, <8 x i8> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x8x4_t %.fca.0.2.insert, <8 x i8> %vld4.fca.3.extract, 0, 3
  ret %struct.int8x8x4_t %.fca.0.3.insert
}

define %struct.int16x4x4_t @test_vld4_s16(i16* readonly %a) {
; CHECK: test_vld4_s16
; CHECK: ld4 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [x{{[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  %vld4 = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8* %1, i32 2)
  %vld4.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int16x4x4_t undef, <4 x i16> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x4_t %.fca.0.0.insert, <4 x i16> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x4_t %.fca.0.1.insert, <4 x i16> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x4x4_t %.fca.0.2.insert, <4 x i16> %vld4.fca.3.extract, 0, 3
  ret %struct.int16x4x4_t %.fca.0.3.insert
}

define %struct.int32x2x4_t @test_vld4_s32(i32* readonly %a) {
; CHECK: test_vld4_s32
; CHECK: ld4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  %vld4 = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4.v2i32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int32x2x4_t undef, <2 x i32> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x4_t %.fca.0.0.insert, <2 x i32> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x4_t %.fca.0.1.insert, <2 x i32> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x2x4_t %.fca.0.2.insert, <2 x i32> %vld4.fca.3.extract, 0, 3
  ret %struct.int32x2x4_t %.fca.0.3.insert
}

define %struct.int64x1x4_t @test_vld4_s64(i64* readonly %a) {
; CHECK: test_vld4_s64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  %vld4 = tail call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.int64x1x4_t undef, <1 x i64> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x4_t %.fca.0.0.insert, <1 x i64> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x4_t %.fca.0.1.insert, <1 x i64> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x1x4_t %.fca.0.2.insert, <1 x i64> %vld4.fca.3.extract, 0, 3
  ret %struct.int64x1x4_t %.fca.0.3.insert
}

define %struct.float32x2x4_t @test_vld4_f32(float* readonly %a) {
; CHECK: test_vld4_f32
; CHECK: ld4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [x{{[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  %vld4 = tail call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4.v2f32(i8* %1, i32 4)
  %vld4.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float32x2x4_t undef, <2 x float> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x4_t %.fca.0.0.insert, <2 x float> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x4_t %.fca.0.1.insert, <2 x float> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x2x4_t %.fca.0.2.insert, <2 x float> %vld4.fca.3.extract, 0, 3
  ret %struct.float32x2x4_t %.fca.0.3.insert
}

define %struct.float64x1x4_t @test_vld4_f64(double* readonly %a) {
; CHECK: test_vld4_f64
; CHECK: ld1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [x{{[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  %vld4 = tail call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8* %1, i32 8)
  %vld4.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.float64x1x4_t undef, <1 x double> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x4_t %.fca.0.0.insert, <1 x double> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x4_t %.fca.0.1.insert, <1 x double> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x1x4_t %.fca.0.2.insert, <1 x double> %vld4.fca.3.extract, 0, 3
  ret %struct.float64x1x4_t %.fca.0.3.insert
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8*, i32)
declare <8 x i16> @llvm.arm.neon.vld1.v8i16(i8*, i32)
declare <4 x i32> @llvm.arm.neon.vld1.v4i32(i8*, i32)
declare <2 x i64> @llvm.arm.neon.vld1.v2i64(i8*, i32)
declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32)
declare <2 x double> @llvm.arm.neon.vld1.v2f64(i8*, i32)
declare <8 x i8> @llvm.arm.neon.vld1.v8i8(i8*, i32)
declare <4 x i16> @llvm.arm.neon.vld1.v4i16(i8*, i32)
declare <2 x i32> @llvm.arm.neon.vld1.v2i32(i8*, i32)
declare <1 x i64> @llvm.arm.neon.vld1.v1i64(i8*, i32)
declare <2 x float> @llvm.arm.neon.vld1.v2f32(i8*, i32)
declare <1 x double> @llvm.arm.neon.vld1.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2.v2i64(i8*, i32)
declare { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8*, i32)
declare { <2 x double>, <2 x double> } @llvm.arm.neon.vld2.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8*, i32)
declare { <2 x float>, <2 x float> } @llvm.arm.neon.vld2.v2f32(i8*, i32)
declare { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4.v16i8(i8*, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4.v8i16(i8*, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32(i8*, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4.v2i64(i8*, i32)
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4.v4f32(i8*, i32)
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4.v2f64(i8*, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4.v8i8(i8*, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4.v4i16(i8*, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4.v2i32(i8*, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4.v2f32(i8*, i32)
declare { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8*, i32)

define void @test_vst1q_s8(i8* %a, <16 x i8> %b) {
; CHECK: test_vst1q_s8
; CHECK: st1 {v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  tail call void @llvm.arm.neon.vst1.v16i8(i8* %a, <16 x i8> %b, i32 1)
  ret void
}

define void @test_vst1q_s16(i16* %a, <8 x i16> %b) {
; CHECK: test_vst1q_s16
; CHECK: st1 {v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %1, <8 x i16> %b, i32 2)
  ret void
}

define void @test_vst1q_s32(i32* %a, <4 x i32> %b) {
; CHECK: test_vst1q_s32
; CHECK: st1 {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4i32(i8* %1, <4 x i32> %b, i32 4)
  ret void
}

define void @test_vst1q_s64(i64* %a, <2 x i64> %b) {
; CHECK: test_vst1q_s64
; CHECK: st1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2i64(i8* %1, <2 x i64> %b, i32 8)
  ret void
}

define void @test_vst1q_f32(float* %a, <4 x float> %b) {
; CHECK: test_vst1q_f32
; CHECK: st1 {v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4f32(i8* %1, <4 x float> %b, i32 4)
  ret void
}

define void @test_vst1q_f64(double* %a, <2 x double> %b) {
; CHECK: test_vst1q_f64
; CHECK: st1 {v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2f64(i8* %1, <2 x double> %b, i32 8)
  ret void
}

define void @test_vst1_s8(i8* %a, <8 x i8> %b) {
; CHECK: test_vst1_s8
; CHECK: st1 {v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  tail call void @llvm.arm.neon.vst1.v8i8(i8* %a, <8 x i8> %b, i32 1)
  ret void
}

define void @test_vst1_s16(i16* %a, <4 x i16> %b) {
; CHECK: test_vst1_s16
; CHECK: st1 {v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst1.v4i16(i8* %1, <4 x i16> %b, i32 2)
  ret void
}

define void @test_vst1_s32(i32* %a, <2 x i32> %b) {
; CHECK: test_vst1_s32
; CHECK: st1 {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2i32(i8* %1, <2 x i32> %b, i32 4)
  ret void
}

define void @test_vst1_s64(i64* %a, <1 x i64> %b) {
; CHECK: test_vst1_s64
; CHECK: st1 {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst1.v1i64(i8* %1, <1 x i64> %b, i32 8)
  ret void
}

define void @test_vst1_f32(float* %a, <2 x float> %b) {
; CHECK: test_vst1_f32
; CHECK: st1 {v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst1.v2f32(i8* %1, <2 x float> %b, i32 4)
  ret void
}

define void @test_vst1_f64(double* %a, <1 x double> %b) {
; CHECK: test_vst1_f64
; CHECK: st1 {v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst1.v1f64(i8* %1, <1 x double> %b, i32 8)
  ret void
}

define void @test_vst2q_s8(i8* %a, [2 x <16 x i8>] %b.coerce) {
; CHECK: test_vst2q_s8
; CHECK: st2 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <16 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, i32 1)
  ret void
}

define void @test_vst2q_s16(i16* %a, [2 x <8 x i16>] %b.coerce) {
; CHECK: test_vst2q_s16
; CHECK: st2 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i16>] %b.coerce, 1
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, i32 2)
  ret void
}

define void @test_vst2q_s32(i32* %a, [2 x <4 x i32>] %b.coerce) {
; CHECK: test_vst2q_s32
; CHECK: st2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %b.coerce, 1
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2q_s64(i64* %a, [2 x <2 x i64>] %b.coerce) {
; CHECK: test_vst2q_s64
; CHECK: st2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i64>] %b.coerce, 1
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2q_f32(float* %a, [2 x <4 x float>] %b.coerce) {
; CHECK: test_vst2q_f32
; CHECK: st2 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x float>] %b.coerce, 1
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2q_f64(double* %a, [2 x <2 x double>] %b.coerce) {
; CHECK: test_vst2q_f64
; CHECK: st2 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x double>] %b.coerce, 1
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2_s8(i8* %a, [2 x <8 x i8>] %b.coerce) {
; CHECK: test_vst2_s8
; CHECK: st2 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, i32 1)
  ret void
}

define void @test_vst2_s16(i16* %a, [2 x <4 x i16>] %b.coerce) {
; CHECK: test_vst2_s16
; CHECK: st2 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i16>] %b.coerce, 1
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, i32 2)
  ret void
}

define void @test_vst2_s32(i32* %a, [2 x <2 x i32>] %b.coerce) {
; CHECK: test_vst2_s32
; CHECK: st2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i32>] %b.coerce, 1
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2_s64(i64* %a, [2 x <1 x i64>] %b.coerce) {
; CHECK: test_vst2_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x i64>] %b.coerce, 1
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst2_f32(float* %a, [2 x <2 x float>] %b.coerce) {
; CHECK: test_vst2_f32
; CHECK: st2 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x float>] %b.coerce, 1
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, i32 4)
  ret void
}

define void @test_vst2_f64(double* %a, [2 x <1 x double>] %b.coerce) {
; CHECK: test_vst2_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x double>] %b.coerce, 1
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, i32 8)
  ret void
}

define void @test_vst3q_s8(i8* %a, [3 x <16 x i8>] %b.coerce) {
; CHECK: test_vst3q_s8
; CHECK: st3 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <16 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, i32 1)
  ret void
}

define void @test_vst3q_s16(i16* %a, [3 x <8 x i16>] %b.coerce) {
; CHECK: test_vst3q_s16
; CHECK: st3 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i16>] %b.coerce, 2
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, i32 2)
  ret void
}

define void @test_vst3q_s32(i32* %a, [3 x <4 x i32>] %b.coerce) {
; CHECK: test_vst3q_s32
; CHECK: st3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i32>] %b.coerce, 2
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3q_s64(i64* %a, [3 x <2 x i64>] %b.coerce) {
; CHECK: test_vst3q_s64
; CHECK: st3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i64>] %b.coerce, 2
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3q_f32(float* %a, [3 x <4 x float>] %b.coerce) {
; CHECK: test_vst3q_f32
; CHECK: st3 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x float>] %b.coerce, 2
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3q_f64(double* %a, [3 x <2 x double>] %b.coerce) {
; CHECK: test_vst3q_f64
; CHECK: st3 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x double>] %b.coerce, 2
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3_s8(i8* %a, [3 x <8 x i8>] %b.coerce) {
; CHECK: test_vst3_s8
; CHECK: st3 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, i32 1)
  ret void
}

define void @test_vst3_s16(i16* %a, [3 x <4 x i16>] %b.coerce) {
; CHECK: test_vst3_s16
; CHECK: st3 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i16>] %b.coerce, 2
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, i32 2)
  ret void
}

define void @test_vst3_s32(i32* %a, [3 x <2 x i32>] %b.coerce) {
; CHECK: test_vst3_s32
; CHECK: st3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i32>] %b.coerce, 2
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3_s64(i64* %a, [3 x <1 x i64>] %b.coerce) {
; CHECK: test_vst3_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x i64>] %b.coerce, 2
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst3_f32(float* %a, [3 x <2 x float>] %b.coerce) {
; CHECK: test_vst3_f32
; CHECK: st3 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x float>] %b.coerce, 2
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, i32 4)
  ret void
}

define void @test_vst3_f64(double* %a, [3 x <1 x double>] %b.coerce) {
; CHECK: test_vst3_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x double>] %b.coerce, 2
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, i32 8)
  ret void
}

define void @test_vst4q_s8(i8* %a, [4 x <16 x i8>] %b.coerce) {
; CHECK: test_vst4q_s8
; CHECK: st4 {v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <16 x i8>] %b.coerce, 3
  tail call void @llvm.arm.neon.vst4.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, <16 x i8> %b.coerce.fca.3.extract, i32 1)
  ret void
}

define void @test_vst4q_s16(i16* %a, [4 x <8 x i16>] %b.coerce) {
; CHECK: test_vst4q_s16
; CHECK: st4 {v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i16>] %b.coerce, 3
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4.v8i16(i8* %1, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, <8 x i16> %b.coerce.fca.3.extract, i32 2)
  ret void
}

define void @test_vst4q_s32(i32* %a, [4 x <4 x i32>] %b.coerce) {
; CHECK: test_vst4q_s32
; CHECK: st4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i32>] %b.coerce, 3
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4i32(i8* %1, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, <4 x i32> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4q_s64(i64* %a, [4 x <2 x i64>] %b.coerce) {
; CHECK: test_vst4q_s64
; CHECK: st4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i64>] %b.coerce, 3
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2i64(i8* %1, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, <2 x i64> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4q_f32(float* %a, [4 x <4 x float>] %b.coerce) {
; CHECK: test_vst4q_f32
; CHECK: st4 {v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x float>] %b.coerce, 3
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4f32(i8* %1, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, <4 x float> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4q_f64(double* %a, [4 x <2 x double>] %b.coerce) {
; CHECK: test_vst4q_f64
; CHECK: st4 {v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x double>] %b.coerce, 3
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2f64(i8* %1, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, <2 x double> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4_s8(i8* %a, [4 x <8 x i8>] %b.coerce) {
; CHECK: test_vst4_s8
; CHECK: st4 {v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i8>] %b.coerce, 3
  tail call void @llvm.arm.neon.vst4.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, <8 x i8> %b.coerce.fca.3.extract, i32 1)
  ret void
}

define void @test_vst4_s16(i16* %a, [4 x <4 x i16>] %b.coerce) {
; CHECK: test_vst4_s16
; CHECK: st4 {v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i16>] %b.coerce, 3
  %1 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4.v4i16(i8* %1, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, <4 x i16> %b.coerce.fca.3.extract, i32 2)
  ret void
}

define void @test_vst4_s32(i32* %a, [4 x <2 x i32>] %b.coerce) {
; CHECK: test_vst4_s32
; CHECK: st4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i32>] %b.coerce, 3
  %1 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2i32(i8* %1, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, <2 x i32> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4_s64(i64* %a, [4 x <1 x i64>] %b.coerce) {
; CHECK: test_vst4_s64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x i64>] %b.coerce, 3
  %1 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4.v1i64(i8* %1, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, <1 x i64> %b.coerce.fca.3.extract, i32 8)
  ret void
}

define void @test_vst4_f32(float* %a, [4 x <2 x float>] %b.coerce) {
; CHECK: test_vst4_f32
; CHECK: st4 {v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x float>] %b.coerce, 3
  %1 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4.v2f32(i8* %1, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, <2 x float> %b.coerce.fca.3.extract, i32 4)
  ret void
}

define void @test_vst4_f64(double* %a, [4 x <1 x double>] %b.coerce) {
; CHECK: test_vst4_f64
; CHECK: st1 {v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d, v{{[0-9]+}}.1d}, [{{x[0-9]+|sp}}]
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x double>] %b.coerce, 3
  %1 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4.v1f64(i8* %1, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, <1 x double> %b.coerce.fca.3.extract, i32 8)
  ret void
}

declare void @llvm.arm.neon.vst1.v16i8(i8*, <16 x i8>, i32)
declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>, i32)
declare void @llvm.arm.neon.vst1.v4i32(i8*, <4 x i32>, i32)
declare void @llvm.arm.neon.vst1.v2i64(i8*, <2 x i64>, i32)
declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32)
declare void @llvm.arm.neon.vst1.v2f64(i8*, <2 x double>, i32)
declare void @llvm.arm.neon.vst1.v8i8(i8*, <8 x i8>, i32)
declare void @llvm.arm.neon.vst1.v4i16(i8*, <4 x i16>, i32)
declare void @llvm.arm.neon.vst1.v2i32(i8*, <2 x i32>, i32)
declare void @llvm.arm.neon.vst1.v1i64(i8*, <1 x i64>, i32)
declare void @llvm.arm.neon.vst1.v2f32(i8*, <2 x float>, i32)
declare void @llvm.arm.neon.vst1.v1f64(i8*, <1 x double>, i32)
declare void @llvm.arm.neon.vst2.v16i8(i8*, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst2.v8i16(i8*, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst2.v4i32(i8*, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst2.v2i64(i8*, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst2.v4f32(i8*, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst2.v2f64(i8*, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst2.v8i8(i8*, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst2.v4i16(i8*, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst2.v2i32(i8*, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst2.v1i64(i8*, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst2.v2f32(i8*, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst2.v1f64(i8*, <1 x double>, <1 x double>, i32)
declare void @llvm.arm.neon.vst3.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst3.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst3.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst3.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst3.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst3.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst3.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst3.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst3.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst3.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, i32)
declare void @llvm.arm.neon.vst4.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare void @llvm.arm.neon.vst4.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32)
declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32)
declare void @llvm.arm.neon.vst4.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i32)
declare void @llvm.arm.neon.vst4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32)
declare void @llvm.arm.neon.vst4.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32)
declare void @llvm.arm.neon.vst4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32)
declare void @llvm.arm.neon.vst4.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32)
declare void @llvm.arm.neon.vst4.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32)
declare void @llvm.arm.neon.vst4.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32)
declare void @llvm.arm.neon.vst4.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32)
declare void @llvm.arm.neon.vst4.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, <1 x double>, i32)