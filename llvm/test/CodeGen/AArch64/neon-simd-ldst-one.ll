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

define <16 x i8> @test_ld_from_poll_v16i8(<16 x i8> %a) {
; CHECK-LABEL: test_ld_from_poll_v16i8
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <16 x i8> %a, <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 2, i8 13, i8 14, i8 15, i8 16>
  ret <16 x i8> %b
}

define <8 x i16> @test_ld_from_poll_v8i16(<8 x i16> %a) {
; CHECK-LABEL: test_ld_from_poll_v8i16
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <8 x i16> %a, <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>
  ret <8 x i16> %b
}

define <4 x i32> @test_ld_from_poll_v4i32(<4 x i32> %a) {
; CHECK-LABEL: test_ld_from_poll_v4i32
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <4 x i32> %a, <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %b
}

define <2 x i64> @test_ld_from_poll_v2i64(<2 x i64> %a) {
; CHECK-LABEL: test_ld_from_poll_v2i64
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <2 x i64> %a, <i64 1, i64 2>
  ret <2 x i64> %b
}

define <4 x float> @test_ld_from_poll_v4f32(<4 x float> %a) {
; CHECK-LABEL: test_ld_from_poll_v4f32
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = fadd <4 x float> %a, <float 1.0, float 2.0, float 3.0, float 4.0>
  ret <4 x float> %b
}

define <2 x double> @test_ld_from_poll_v2f64(<2 x double> %a) {
; CHECK-LABEL: test_ld_from_poll_v2f64
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{q[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = fadd <2 x double> %a, <double 1.0, double 2.0>
  ret <2 x double> %b
}

define <8 x i8> @test_ld_from_poll_v8i8(<8 x i8> %a) {
; CHECK-LABEL: test_ld_from_poll_v8i8
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{d[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <8 x i8> %a, <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>
  ret <8 x i8> %b
}

define <4 x i16> @test_ld_from_poll_v4i16(<4 x i16> %a) {
; CHECK-LABEL: test_ld_from_poll_v4i16
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{d[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <4 x i16> %a, <i16 1, i16 2, i16 3, i16 4>
  ret <4 x i16> %b
}

define <2 x i32> @test_ld_from_poll_v2i32(<2 x i32> %a) {
; CHECK-LABEL: test_ld_from_poll_v2i32
; CHECK: adrp {{x[0-9]+}}, .{{[A-Z0-9_]+}}
; CHECK-NEXT: ldr {{d[0-9]+}}, [{{x[0-9]+}}, #:lo12:.{{[A-Z0-9_]+}}]
entry:
  %b = add <2 x i32> %a, <i32 1, i32 2>
  ret <2 x i32> %b
}

define <16 x i8> @test_vld1q_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld1q_dup_s8
; CHECK: ld1r {{{v[0-9]+}}.16b}, [x0]
entry:
  %0 = load i8* %a, align 1
  %1 = insertelement <16 x i8> undef, i8 %0, i32 0
  %lane = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %lane
}

define <8 x i16> @test_vld1q_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld1q_dup_s16
; CHECK: ld1r {{{v[0-9]+}}.8h}, [x0]
entry:
  %0 = load i16* %a, align 2
  %1 = insertelement <8 x i16> undef, i16 %0, i32 0
  %lane = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %lane
}

define <4 x i32> @test_vld1q_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld1q_dup_s32
; CHECK: ld1r {{{v[0-9]+}}.4s}, [x0]
entry:
  %0 = load i32* %a, align 4
  %1 = insertelement <4 x i32> undef, i32 %0, i32 0
  %lane = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %lane
}

define <2 x i64> @test_vld1q_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld1q_dup_s64
; CHECK: ld1r {{{v[0-9]+}}.2d}, [x0]
entry:
  %0 = load i64* %a, align 8
  %1 = insertelement <2 x i64> undef, i64 %0, i32 0
  %lane = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %lane
}

define <4 x float> @test_vld1q_dup_f32(float* %a) {
; CHECK-LABEL: test_vld1q_dup_f32
; CHECK: ld1r {{{v[0-9]+}}.4s}, [x0]
entry:
  %0 = load float* %a, align 4
  %1 = insertelement <4 x float> undef, float %0, i32 0
  %lane = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %lane
}

define <2 x double> @test_vld1q_dup_f64(double* %a) {
; CHECK-LABEL: test_vld1q_dup_f64
; CHECK: ld1r {{{v[0-9]+}}.2d}, [x0]
entry:
  %0 = load double* %a, align 8
  %1 = insertelement <2 x double> undef, double %0, i32 0
  %lane = shufflevector <2 x double> %1, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %lane
}

define <8 x i8> @test_vld1_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld1_dup_s8
; CHECK: ld1r {{{v[0-9]+}}.8b}, [x0]
entry:
  %0 = load i8* %a, align 1
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %lane = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  ret <8 x i8> %lane
}

define <4 x i16> @test_vld1_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld1_dup_s16
; CHECK: ld1r {{{v[0-9]+}}.4h}, [x0]
entry:
  %0 = load i16* %a, align 2
  %1 = insertelement <4 x i16> undef, i16 %0, i32 0
  %lane = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> zeroinitializer
  ret <4 x i16> %lane
}

define <2 x i32> @test_vld1_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld1_dup_s32
; CHECK: ld1r {{{v[0-9]+}}.2s}, [x0]
entry:
  %0 = load i32* %a, align 4
  %1 = insertelement <2 x i32> undef, i32 %0, i32 0
  %lane = shufflevector <2 x i32> %1, <2 x i32> undef, <2 x i32> zeroinitializer
  ret <2 x i32> %lane
}

define <1 x i64> @test_vld1_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld1_dup_s64
; CHECK: ld1r {{{v[0-9]+}}.1d}, [x0]
entry:
  %0 = load i64* %a, align 8
  %1 = insertelement <1 x i64> undef, i64 %0, i32 0
  ret <1 x i64> %1
}

define <2 x float> @test_vld1_dup_f32(float* %a) {
; CHECK-LABEL: test_vld1_dup_f32
; CHECK: ld1r {{{v[0-9]+}}.2s}, [x0]
entry:
  %0 = load float* %a, align 4
  %1 = insertelement <2 x float> undef, float %0, i32 0
  %lane = shufflevector <2 x float> %1, <2 x float> undef, <2 x i32> zeroinitializer
  ret <2 x float> %lane
}

define <1 x double> @test_vld1_dup_f64(double* %a) {
; CHECK-LABEL: test_vld1_dup_f64
; CHECK: ld1r {{{v[0-9]+}}.1d}, [x0]
entry:
  %0 = load double* %a, align 8
  %1 = insertelement <1 x double> undef, double %0, i32 0
  ret <1 x double> %1
}

define %struct.int8x16x2_t @test_vld2q_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld2q_dup_s8
; CHECK: ld2r {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, [x0]
entry:
  %vld_dup = tail call { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2lane.v16i8(i8* %a, <16 x i8> undef, <16 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <16 x i8>, <16 x i8> } %vld_dup, 0
  %lane = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %1 = extractvalue { <16 x i8>, <16 x i8> } %vld_dup, 1
  %lane1 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x16x2_t undef, <16 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x2_t %.fca.0.0.insert, <16 x i8> %lane1, 0, 1
  ret %struct.int8x16x2_t %.fca.0.1.insert
}

define %struct.int16x8x2_t @test_vld2q_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld2q_dup_s16
; CHECK: ld2r {{{v[0-9]+}}.8h, {{v[0-9]+}}.8h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2lane.v8i16(i8* %0, <8 x i16> undef, <8 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <8 x i16>, <8 x i16> } %vld_dup, 0
  %lane = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> zeroinitializer
  %2 = extractvalue { <8 x i16>, <8 x i16> } %vld_dup, 1
  %lane1 = shufflevector <8 x i16> %2, <8 x i16> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %lane1, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vld2q_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld2q_dup_s32
; CHECK: ld2r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8* %0, <4 x i32> undef, <4 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x i32>, <4 x i32> } %vld_dup, 0
  %lane = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i32>, <4 x i32> } %vld_dup, 1
  %lane1 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %lane1, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.int64x2x2_t @test_vld2q_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld2q_dup_s64
; CHECK: ld2r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2lane.v2i64(i8* %0, <2 x i64> undef, <2 x i64> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x i64>, <2 x i64> } %vld_dup, 0
  %lane = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i64>, <2 x i64> } %vld_dup, 1
  %lane1 = shufflevector <2 x i64> %2, <2 x i64> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int64x2x2_t undef, <2 x i64> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x2_t %.fca.0.0.insert, <2 x i64> %lane1, 0, 1
  ret %struct.int64x2x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vld2q_dup_f32(float* %a) {
; CHECK-LABEL: test_vld2q_dup_f32
; CHECK: ld2r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2lane.v4f32(i8* %0, <4 x float> undef, <4 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x float>, <4 x float> } %vld_dup, 0
  %lane = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x float>, <4 x float> } %vld_dup, 1
  %lane1 = shufflevector <4 x float> %2, <4 x float> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %lane1, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.float64x2x2_t @test_vld2q_dup_f64(double* %a) {
; CHECK-LABEL: test_vld2q_dup_f64
; CHECK: ld2r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <2 x double>, <2 x double> } @llvm.arm.neon.vld2lane.v2f64(i8* %0, <2 x double> undef, <2 x double> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x double>, <2 x double> } %vld_dup, 0
  %lane = shufflevector <2 x double> %1, <2 x double> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x double>, <2 x double> } %vld_dup, 1
  %lane1 = shufflevector <2 x double> %2, <2 x double> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float64x2x2_t undef, <2 x double> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x2_t %.fca.0.0.insert, <2 x double> %lane1, 0, 1
  ret %struct.float64x2x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vld2_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld2_dup_s8
; CHECK: ld2r {{{v[0-9]+}}.8b, {{v[0-9]+}}.8b}, [x0]
entry:
  %vld_dup = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8* %a, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <8 x i8>, <8 x i8> } %vld_dup, 0
  %lane = shufflevector <8 x i8> %0, <8 x i8> undef, <8 x i32> zeroinitializer
  %1 = extractvalue { <8 x i8>, <8 x i8> } %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %lane1, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vld2_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld2_dup_s16
; CHECK: ld2r {{{v[0-9]+}}.4h, {{v[0-9]+}}.4h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2lane.v4i16(i8* %0, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <4 x i16>, <4 x i16> } %vld_dup, 0
  %lane = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i16>, <4 x i16> } %vld_dup, 1
  %lane1 = shufflevector <4 x i16> %2, <4 x i16> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %lane1, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vld2_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld2_dup_s32
; CHECK: ld2r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2lane.v2i32(i8* %0, <2 x i32> undef, <2 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x i32>, <2 x i32> } %vld_dup, 0
  %lane = shufflevector <2 x i32> %1, <2 x i32> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i32>, <2 x i32> } %vld_dup, 1
  %lane1 = shufflevector <2 x i32> %2, <2 x i32> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %lane1, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.int64x1x2_t @test_vld2_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld2_dup_s64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld_dup, 1
  %.fca.0.0.insert = insertvalue %struct.int64x1x2_t undef, <1 x i64> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x2_t %.fca.0.0.insert, <1 x i64> %vld_dup.fca.1.extract, 0, 1
  ret %struct.int64x1x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vld2_dup_f32(float* %a) {
; CHECK-LABEL: test_vld2_dup_f32
; CHECK: ld2r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2lane.v2f32(i8* %0, <2 x float> undef, <2 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x float>, <2 x float> } %vld_dup, 0
  %lane = shufflevector <2 x float> %1, <2 x float> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x float>, <2 x float> } %vld_dup, 1
  %lane1 = shufflevector <2 x float> %2, <2 x float> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %lane1, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.float64x1x2_t @test_vld2_dup_f64(double* %a) {
; CHECK-LABEL: test_vld2_dup_f64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x double>, <1 x double> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x double>, <1 x double> } %vld_dup, 1
  %.fca.0.0.insert = insertvalue %struct.float64x1x2_t undef, <1 x double> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x2_t %.fca.0.0.insert, <1 x double> %vld_dup.fca.1.extract, 0, 1
  ret %struct.float64x1x2_t %.fca.0.1.insert
}

define %struct.int8x16x3_t @test_vld3q_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld3q_dup_s8
; CHECK: ld3r {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, [x0]
entry:
  %vld_dup = tail call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3lane.v16i8(i8* %a, <16 x i8> undef, <16 x i8> undef, <16 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 0
  %lane = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 1
  %lane1 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> zeroinitializer
  %2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 2
  %lane2 = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x16x3_t undef, <16 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x3_t %.fca.0.0.insert, <16 x i8> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x3_t %.fca.0.1.insert, <16 x i8> %lane2, 0, 2
  ret %struct.int8x16x3_t %.fca.0.2.insert
}

define %struct.int16x8x3_t @test_vld3q_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld3q_dup_s16
; CHECK: ld3r {{{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8* %0, <8 x i16> undef, <8 x i16> undef, <8 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 0
  %lane = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> zeroinitializer
  %2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 1
  %lane1 = shufflevector <8 x i16> %2, <8 x i16> undef, <8 x i32> zeroinitializer
  %3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 2
  %lane2 = shufflevector <8 x i16> %3, <8 x i16> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x8x3_t undef, <8 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x3_t %.fca.0.0.insert, <8 x i16> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x3_t %.fca.0.1.insert, <8 x i16> %lane2, 0, 2
  ret %struct.int16x8x3_t %.fca.0.2.insert
}

define %struct.int32x4x3_t @test_vld3q_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld3q_dup_s32
; CHECK: ld3r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8* %0, <4 x i32> undef, <4 x i32> undef, <4 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 0
  %lane = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 1
  %lane1 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 2
  %lane2 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x4x3_t undef, <4 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x3_t %.fca.0.0.insert, <4 x i32> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x3_t %.fca.0.1.insert, <4 x i32> %lane2, 0, 2
  ret %struct.int32x4x3_t %.fca.0.2.insert
}

define %struct.int64x2x3_t @test_vld3q_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld3q_dup_s64
; CHECK: ld3r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3lane.v2i64(i8* %0, <2 x i64> undef, <2 x i64> undef, <2 x i64> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 0
  %lane = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 1
  %lane1 = shufflevector <2 x i64> %2, <2 x i64> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 2
  %lane2 = shufflevector <2 x i64> %3, <2 x i64> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int64x2x3_t undef, <2 x i64> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x3_t %.fca.0.0.insert, <2 x i64> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x3_t %.fca.0.1.insert, <2 x i64> %lane2, 0, 2
  ret %struct.int64x2x3_t %.fca.0.2.insert
}

define %struct.float32x4x3_t @test_vld3q_dup_f32(float* %a) {
; CHECK-LABEL: test_vld3q_dup_f32
; CHECK: ld3r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3lane.v4f32(i8* %0, <4 x float> undef, <4 x float> undef, <4 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld_dup, 0
  %lane = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld_dup, 1
  %lane1 = shufflevector <4 x float> %2, <4 x float> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld_dup, 2
  %lane2 = shufflevector <4 x float> %3, <4 x float> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x4x3_t undef, <4 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x3_t %.fca.0.0.insert, <4 x float> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x3_t %.fca.0.1.insert, <4 x float> %lane2, 0, 2
  ret %struct.float32x4x3_t %.fca.0.2.insert
}

define %struct.float64x2x3_t @test_vld3q_dup_f64(double* %a) {
; CHECK-LABEL: test_vld3q_dup_f64
; CHECK: ld3r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3lane.v2f64(i8* %0, <2 x double> undef, <2 x double> undef, <2 x double> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld_dup, 0
  %lane = shufflevector <2 x double> %1, <2 x double> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld_dup, 1
  %lane1 = shufflevector <2 x double> %2, <2 x double> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld_dup, 2
  %lane2 = shufflevector <2 x double> %3, <2 x double> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float64x2x3_t undef, <2 x double> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x3_t %.fca.0.0.insert, <2 x double> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x3_t %.fca.0.1.insert, <2 x double> %lane2, 0, 2
  ret %struct.float64x2x3_t %.fca.0.2.insert
}

define %struct.int8x8x3_t @test_vld3_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld3_dup_s8
; CHECK: ld3r {{{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b}, [x0]
entry:
  %vld_dup = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3lane.v8i8(i8* %a, <8 x i8> undef, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 0
  %lane = shufflevector <8 x i8> %0, <8 x i8> undef, <8 x i32> zeroinitializer
  %1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 2
  %lane2 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x8x3_t undef, <8 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x3_t %.fca.0.0.insert, <8 x i8> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x3_t %.fca.0.1.insert, <8 x i8> %lane2, 0, 2
  ret %struct.int8x8x3_t %.fca.0.2.insert
}

define %struct.int16x4x3_t @test_vld3_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld3_dup_s16
; CHECK: ld3r {{{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8* %0, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 0
  %lane = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 1
  %lane1 = shufflevector <4 x i16> %2, <4 x i16> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 2
  %lane2 = shufflevector <4 x i16> %3, <4 x i16> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x4x3_t undef, <4 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x3_t %.fca.0.0.insert, <4 x i16> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x3_t %.fca.0.1.insert, <4 x i16> %lane2, 0, 2
  ret %struct.int16x4x3_t %.fca.0.2.insert
}

define %struct.int32x2x3_t @test_vld3_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld3_dup_s32
; CHECK: ld3r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3lane.v2i32(i8* %0, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 0
  %lane = shufflevector <2 x i32> %1, <2 x i32> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 1
  %lane1 = shufflevector <2 x i32> %2, <2 x i32> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 2
  %lane2 = shufflevector <2 x i32> %3, <2 x i32> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x2x3_t undef, <2 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x3_t %.fca.0.0.insert, <2 x i32> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x3_t %.fca.0.1.insert, <2 x i32> %lane2, 0, 2
  ret %struct.int32x2x3_t %.fca.0.2.insert
}

define %struct.int64x1x3_t @test_vld3_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld3_dup_s64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 1
  %vld_dup.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 2
  %.fca.0.0.insert = insertvalue %struct.int64x1x3_t undef, <1 x i64> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x3_t %.fca.0.0.insert, <1 x i64> %vld_dup.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x3_t %.fca.0.1.insert, <1 x i64> %vld_dup.fca.2.extract, 0, 2
  ret %struct.int64x1x3_t %.fca.0.2.insert
}

define %struct.float32x2x3_t @test_vld3_dup_f32(float* %a) {
; CHECK-LABEL: test_vld3_dup_f32
; CHECK: ld3r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3lane.v2f32(i8* %0, <2 x float> undef, <2 x float> undef, <2 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld_dup, 0
  %lane = shufflevector <2 x float> %1, <2 x float> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld_dup, 1
  %lane1 = shufflevector <2 x float> %2, <2 x float> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld_dup, 2
  %lane2 = shufflevector <2 x float> %3, <2 x float> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x2x3_t undef, <2 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x3_t %.fca.0.0.insert, <2 x float> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x3_t %.fca.0.1.insert, <2 x float> %lane2, 0, 2
  ret %struct.float32x2x3_t %.fca.0.2.insert
}

define %struct.float64x1x3_t @test_vld3_dup_f64(double* %a) {
; CHECK-LABEL: test_vld3_dup_f64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld_dup, 1
  %vld_dup.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld_dup, 2
  %.fca.0.0.insert = insertvalue %struct.float64x1x3_t undef, <1 x double> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x3_t %.fca.0.0.insert, <1 x double> %vld_dup.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x3_t %.fca.0.1.insert, <1 x double> %vld_dup.fca.2.extract, 0, 2
  ret %struct.float64x1x3_t %.fca.0.2.insert
}

define %struct.int8x16x4_t @test_vld4q_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld4q_dup_s8
; CHECK: ld4r {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, [x0]
entry:
  %vld_dup = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4lane.v16i8(i8* %a, <16 x i8> undef, <16 x i8> undef, <16 x i8> undef, <16 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 0
  %lane = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 1
  %lane1 = shufflevector <16 x i8> %1, <16 x i8> undef, <16 x i32> zeroinitializer
  %2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 2
  %lane2 = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  %3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld_dup, 3
  %lane3 = shufflevector <16 x i8> %3, <16 x i8> undef, <16 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x16x4_t undef, <16 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x4_t %.fca.0.0.insert, <16 x i8> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x4_t %.fca.0.1.insert, <16 x i8> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x16x4_t %.fca.0.2.insert, <16 x i8> %lane3, 0, 3
  ret %struct.int8x16x4_t %.fca.0.3.insert
}

define %struct.int16x8x4_t @test_vld4q_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld4q_dup_s16
; CHECK: ld4r {{{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4lane.v8i16(i8* %0, <8 x i16> undef, <8 x i16> undef, <8 x i16> undef, <8 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 0
  %lane = shufflevector <8 x i16> %1, <8 x i16> undef, <8 x i32> zeroinitializer
  %2 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 1
  %lane1 = shufflevector <8 x i16> %2, <8 x i16> undef, <8 x i32> zeroinitializer
  %3 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 2
  %lane2 = shufflevector <8 x i16> %3, <8 x i16> undef, <8 x i32> zeroinitializer
  %4 = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld_dup, 3
  %lane3 = shufflevector <8 x i16> %4, <8 x i16> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x8x4_t undef, <8 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x4_t %.fca.0.0.insert, <8 x i16> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x4_t %.fca.0.1.insert, <8 x i16> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x8x4_t %.fca.0.2.insert, <8 x i16> %lane3, 0, 3
  ret %struct.int16x8x4_t %.fca.0.3.insert
}

define %struct.int32x4x4_t @test_vld4q_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld4q_dup_s32
; CHECK: ld4r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8* %0, <4 x i32> undef, <4 x i32> undef, <4 x i32> undef, <4 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 0
  %lane = shufflevector <4 x i32> %1, <4 x i32> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 1
  %lane1 = shufflevector <4 x i32> %2, <4 x i32> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 2
  %lane2 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> zeroinitializer
  %4 = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld_dup, 3
  %lane3 = shufflevector <4 x i32> %4, <4 x i32> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x4x4_t undef, <4 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x4_t %.fca.0.0.insert, <4 x i32> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x4_t %.fca.0.1.insert, <4 x i32> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x4x4_t %.fca.0.2.insert, <4 x i32> %lane3, 0, 3
  ret %struct.int32x4x4_t %.fca.0.3.insert
}

define %struct.int64x2x4_t @test_vld4q_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld4q_dup_s64
; CHECK: ld4r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4lane.v2i64(i8* %0, <2 x i64> undef, <2 x i64> undef, <2 x i64> undef, <2 x i64> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 0
  %lane = shufflevector <2 x i64> %1, <2 x i64> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 1
  %lane1 = shufflevector <2 x i64> %2, <2 x i64> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 2
  %lane2 = shufflevector <2 x i64> %3, <2 x i64> undef, <2 x i32> zeroinitializer
  %4 = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld_dup, 3
  %lane3 = shufflevector <2 x i64> %4, <2 x i64> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int64x2x4_t undef, <2 x i64> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x4_t %.fca.0.0.insert, <2 x i64> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x4_t %.fca.0.1.insert, <2 x i64> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x2x4_t %.fca.0.2.insert, <2 x i64> %lane3, 0, 3
  ret %struct.int64x2x4_t %.fca.0.3.insert
}

define %struct.float32x4x4_t @test_vld4q_dup_f32(float* %a) {
; CHECK-LABEL: test_vld4q_dup_f32
; CHECK: ld4r {{{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4lane.v4f32(i8* %0, <4 x float> undef, <4 x float> undef, <4 x float> undef, <4 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld_dup, 0
  %lane = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld_dup, 1
  %lane1 = shufflevector <4 x float> %2, <4 x float> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld_dup, 2
  %lane2 = shufflevector <4 x float> %3, <4 x float> undef, <4 x i32> zeroinitializer
  %4 = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld_dup, 3
  %lane3 = shufflevector <4 x float> %4, <4 x float> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x4x4_t undef, <4 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x4_t %.fca.0.0.insert, <4 x float> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x4_t %.fca.0.1.insert, <4 x float> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x4x4_t %.fca.0.2.insert, <4 x float> %lane3, 0, 3
  ret %struct.float32x4x4_t %.fca.0.3.insert
}

define %struct.float64x2x4_t @test_vld4q_dup_f64(double* %a) {
; CHECK-LABEL: test_vld4q_dup_f64
; CHECK: ld4r {{{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8* %0, <2 x double> undef, <2 x double> undef, <2 x double> undef, <2 x double> undef, i32 0, i32 8)
  %1 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld_dup, 0
  %lane = shufflevector <2 x double> %1, <2 x double> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld_dup, 1
  %lane1 = shufflevector <2 x double> %2, <2 x double> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld_dup, 2
  %lane2 = shufflevector <2 x double> %3, <2 x double> undef, <2 x i32> zeroinitializer
  %4 = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld_dup, 3
  %lane3 = shufflevector <2 x double> %4, <2 x double> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float64x2x4_t undef, <2 x double> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x4_t %.fca.0.0.insert, <2 x double> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x4_t %.fca.0.1.insert, <2 x double> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x2x4_t %.fca.0.2.insert, <2 x double> %lane3, 0, 3
  ret %struct.float64x2x4_t %.fca.0.3.insert
}

define %struct.int8x8x4_t @test_vld4_dup_s8(i8* %a) {
; CHECK-LABEL: test_vld4_dup_s8
; CHECK: ld4r {{{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b}, [x0]
entry:
  %vld_dup = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4lane.v8i8(i8* %a, <8 x i8> undef, <8 x i8> undef, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %0 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 0
  %lane = shufflevector <8 x i8> %0, <8 x i8> undef, <8 x i32> zeroinitializer
  %1 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %2 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 2
  %lane2 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %3 = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld_dup, 3
  %lane3 = shufflevector <8 x i8> %3, <8 x i8> undef, <8 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int8x8x4_t undef, <8 x i8> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x4_t %.fca.0.0.insert, <8 x i8> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x4_t %.fca.0.1.insert, <8 x i8> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x8x4_t %.fca.0.2.insert, <8 x i8> %lane3, 0, 3
  ret %struct.int8x8x4_t %.fca.0.3.insert
}

define %struct.int16x4x4_t @test_vld4_dup_s16(i16* %a) {
; CHECK-LABEL: test_vld4_dup_s16
; CHECK: ld4r {{{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h}, [x0]
entry:
  %0 = bitcast i16* %a to i8*
  %vld_dup = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4lane.v4i16(i8* %0, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
  %1 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 0
  %lane = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> zeroinitializer
  %2 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 1
  %lane1 = shufflevector <4 x i16> %2, <4 x i16> undef, <4 x i32> zeroinitializer
  %3 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 2
  %lane2 = shufflevector <4 x i16> %3, <4 x i16> undef, <4 x i32> zeroinitializer
  %4 = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld_dup, 3
  %lane3 = shufflevector <4 x i16> %4, <4 x i16> undef, <4 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int16x4x4_t undef, <4 x i16> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x4_t %.fca.0.0.insert, <4 x i16> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x4_t %.fca.0.1.insert, <4 x i16> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x4x4_t %.fca.0.2.insert, <4 x i16> %lane3, 0, 3
  ret %struct.int16x4x4_t %.fca.0.3.insert
}

define %struct.int32x2x4_t @test_vld4_dup_s32(i32* %a) {
; CHECK-LABEL: test_vld4_dup_s32
; CHECK: ld4r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast i32* %a to i8*
  %vld_dup = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8* %0, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 0
  %lane = shufflevector <2 x i32> %1, <2 x i32> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 1
  %lane1 = shufflevector <2 x i32> %2, <2 x i32> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 2
  %lane2 = shufflevector <2 x i32> %3, <2 x i32> undef, <2 x i32> zeroinitializer
  %4 = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld_dup, 3
  %lane3 = shufflevector <2 x i32> %4, <2 x i32> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.int32x2x4_t undef, <2 x i32> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x4_t %.fca.0.0.insert, <2 x i32> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x4_t %.fca.0.1.insert, <2 x i32> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x2x4_t %.fca.0.2.insert, <2 x i32> %lane3, 0, 3
  ret %struct.int32x2x4_t %.fca.0.3.insert
}

define %struct.int64x1x4_t @test_vld4_dup_s64(i64* %a) {
; CHECK-LABEL: test_vld4_dup_s64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast i64* %a to i8*
  %vld_dup = tail call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 1
  %vld_dup.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 2
  %vld_dup.fca.3.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld_dup, 3
  %.fca.0.0.insert = insertvalue %struct.int64x1x4_t undef, <1 x i64> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x4_t %.fca.0.0.insert, <1 x i64> %vld_dup.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x4_t %.fca.0.1.insert, <1 x i64> %vld_dup.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x1x4_t %.fca.0.2.insert, <1 x i64> %vld_dup.fca.3.extract, 0, 3
  ret %struct.int64x1x4_t %.fca.0.3.insert
}

define %struct.float32x2x4_t @test_vld4_dup_f32(float* %a) {
; CHECK-LABEL: test_vld4_dup_f32
; CHECK: ld4r {{{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s}, [x0]
entry:
  %0 = bitcast float* %a to i8*
  %vld_dup = tail call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4lane.v2f32(i8* %0, <2 x float> undef, <2 x float> undef, <2 x float> undef, <2 x float> undef, i32 0, i32 4)
  %1 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld_dup, 0
  %lane = shufflevector <2 x float> %1, <2 x float> undef, <2 x i32> zeroinitializer
  %2 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld_dup, 1
  %lane1 = shufflevector <2 x float> %2, <2 x float> undef, <2 x i32> zeroinitializer
  %3 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld_dup, 2
  %lane2 = shufflevector <2 x float> %3, <2 x float> undef, <2 x i32> zeroinitializer
  %4 = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld_dup, 3
  %lane3 = shufflevector <2 x float> %4, <2 x float> undef, <2 x i32> zeroinitializer
  %.fca.0.0.insert = insertvalue %struct.float32x2x4_t undef, <2 x float> %lane, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x4_t %.fca.0.0.insert, <2 x float> %lane1, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x4_t %.fca.0.1.insert, <2 x float> %lane2, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x2x4_t %.fca.0.2.insert, <2 x float> %lane3, 0, 3
  ret %struct.float32x2x4_t %.fca.0.3.insert
}

define %struct.float64x1x4_t @test_vld4_dup_f64(double* %a) {
; CHECK-LABEL: test_vld4_dup_f64
; CHECK: ld1 {{{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d}, [x0]
entry:
  %0 = bitcast double* %a to i8*
  %vld_dup = tail call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8* %0, i32 8)
  %vld_dup.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld_dup, 0
  %vld_dup.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld_dup, 1
  %vld_dup.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld_dup, 2
  %vld_dup.fca.3.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld_dup, 3
  %.fca.0.0.insert = insertvalue %struct.float64x1x4_t undef, <1 x double> %vld_dup.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x4_t %.fca.0.0.insert, <1 x double> %vld_dup.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x4_t %.fca.0.1.insert, <1 x double> %vld_dup.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x1x4_t %.fca.0.2.insert, <1 x double> %vld_dup.fca.3.extract, 0, 3
  ret %struct.float64x1x4_t %.fca.0.3.insert
}

define <16 x i8> @test_vld1q_lane_s8(i8* %a, <16 x i8> %b) {
; CHECK-LABEL: test_vld1q_lane_s8
; CHECK: ld1 {{{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %0 = load i8* %a, align 1
  %vld1_lane = insertelement <16 x i8> %b, i8 %0, i32 15
  ret <16 x i8> %vld1_lane
}

define <8 x i16> @test_vld1q_lane_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vld1q_lane_s16
; CHECK: ld1 {{{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %0 = load i16* %a, align 2
  %vld1_lane = insertelement <8 x i16> %b, i16 %0, i32 7
  ret <8 x i16> %vld1_lane
}

define <4 x i32> @test_vld1q_lane_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vld1q_lane_s32
; CHECK: ld1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = load i32* %a, align 4
  %vld1_lane = insertelement <4 x i32> %b, i32 %0, i32 3
  ret <4 x i32> %vld1_lane
}

define <2 x i64> @test_vld1q_lane_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vld1q_lane_s64
; CHECK: ld1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = load i64* %a, align 8
  %vld1_lane = insertelement <2 x i64> %b, i64 %0, i32 1
  ret <2 x i64> %vld1_lane
}

define <4 x float> @test_vld1q_lane_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vld1q_lane_f32
; CHECK: ld1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = load float* %a, align 4
  %vld1_lane = insertelement <4 x float> %b, float %0, i32 3
  ret <4 x float> %vld1_lane
}

define <2 x double> @test_vld1q_lane_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vld1q_lane_f64
; CHECK: ld1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = load double* %a, align 8
  %vld1_lane = insertelement <2 x double> %b, double %0, i32 1
  ret <2 x double> %vld1_lane
}

define <8 x i8> @test_vld1_lane_s8(i8* %a, <8 x i8> %b) {
; CHECK-LABEL: test_vld1_lane_s8
; CHECK: ld1 {{{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %0 = load i8* %a, align 1
  %vld1_lane = insertelement <8 x i8> %b, i8 %0, i32 7
  ret <8 x i8> %vld1_lane
}

define <4 x i16> @test_vld1_lane_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vld1_lane_s16
; CHECK: ld1 {{{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %0 = load i16* %a, align 2
  %vld1_lane = insertelement <4 x i16> %b, i16 %0, i32 3
  ret <4 x i16> %vld1_lane
}

define <2 x i32> @test_vld1_lane_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vld1_lane_s32
; CHECK: ld1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = load i32* %a, align 4
  %vld1_lane = insertelement <2 x i32> %b, i32 %0, i32 1
  ret <2 x i32> %vld1_lane
}

define <1 x i64> @test_vld1_lane_s64(i64* %a, <1 x i64> %b) {
; CHECK-LABEL: test_vld1_lane_s64
; CHECK: ld1r {{{v[0-9]+}}.1d}, [x0]
entry:
  %0 = load i64* %a, align 8
  %vld1_lane = insertelement <1 x i64> undef, i64 %0, i32 0
  ret <1 x i64> %vld1_lane
}

define <2 x float> @test_vld1_lane_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vld1_lane_f32
; CHECK: ld1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = load float* %a, align 4
  %vld1_lane = insertelement <2 x float> %b, float %0, i32 1
  ret <2 x float> %vld1_lane
}

define <1 x double> @test_vld1_lane_f64(double* %a, <1 x double> %b) {
; CHECK-LABEL: test_vld1_lane_f64
; CHECK: ld1r {{{v[0-9]+}}.1d}, [x0]
entry:
  %0 = load double* %a, align 8
  %vld1_lane = insertelement <1 x double> undef, double %0, i32 0
  ret <1 x double> %vld1_lane
}

define %struct.int16x8x2_t @test_vld2q_lane_s16(i16* %a, [2 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld2q_lane_s16
; CHECK: ld2 {{{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i16>] %b.coerce, 1
  %0 = bitcast i16* %a to i8*
  %vld2_lane = tail call { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, i32 7, i32 2)
  %vld2_lane.fca.0.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <8 x i16>, <8 x i16> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int16x8x2_t undef, <8 x i16> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x2_t %.fca.0.0.insert, <8 x i16> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int16x8x2_t %.fca.0.1.insert
}

define %struct.int32x4x2_t @test_vld2q_lane_s32(i32* %a, [2 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld2q_lane_s32
; CHECK: ld2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %b.coerce, 1
  %0 = bitcast i32* %a to i8*
  %vld2_lane = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, i32 3, i32 4)
  %vld2_lane.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int32x4x2_t undef, <4 x i32> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x2_t %.fca.0.0.insert, <4 x i32> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int32x4x2_t %.fca.0.1.insert
}

define %struct.int64x2x2_t @test_vld2q_lane_s64(i64* %a, [2 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld2q_lane_s64
; CHECK: ld2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i64>] %b.coerce, 1
  %0 = bitcast i64* %a to i8*
  %vld2_lane = tail call { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, i32 1, i32 8)
  %vld2_lane.fca.0.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <2 x i64>, <2 x i64> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int64x2x2_t undef, <2 x i64> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x2_t %.fca.0.0.insert, <2 x i64> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int64x2x2_t %.fca.0.1.insert
}

define %struct.float32x4x2_t @test_vld2q_lane_f32(float* %a, [2 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vld2q_lane_f32
; CHECK: ld2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x float>] %b.coerce, 1
  %0 = bitcast float* %a to i8*
  %vld2_lane = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, i32 3, i32 4)
  %vld2_lane.fca.0.extract = extractvalue { <4 x float>, <4 x float> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <4 x float>, <4 x float> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.float32x4x2_t undef, <4 x float> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x2_t %.fca.0.0.insert, <4 x float> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.float32x4x2_t %.fca.0.1.insert
}

define %struct.float64x2x2_t @test_vld2q_lane_f64(double* %a, [2 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vld2q_lane_f64
; CHECK: ld2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x double>] %b.coerce, 1
  %0 = bitcast double* %a to i8*
  %vld2_lane = tail call { <2 x double>, <2 x double> } @llvm.arm.neon.vld2lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, i32 1, i32 8)
  %vld2_lane.fca.0.extract = extractvalue { <2 x double>, <2 x double> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <2 x double>, <2 x double> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.float64x2x2_t undef, <2 x double> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x2_t %.fca.0.0.insert, <2 x double> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.float64x2x2_t %.fca.0.1.insert
}

define %struct.int8x8x2_t @test_vld2_lane_s8(i8* %a, [2 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_s8
; CHECK: ld2 {{{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i8>] %b.coerce, 1
  %vld2_lane = tail call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, i32 7, i32 1)
  %vld2_lane.fca.0.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <8 x i8>, <8 x i8> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int8x8x2_t undef, <8 x i8> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x2_t %.fca.0.0.insert, <8 x i8> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int8x8x2_t %.fca.0.1.insert
}

define %struct.int16x4x2_t @test_vld2_lane_s16(i16* %a, [2 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_s16
; CHECK: ld2 {{{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i16>] %b.coerce, 1
  %0 = bitcast i16* %a to i8*
  %vld2_lane = tail call { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, i32 3, i32 2)
  %vld2_lane.fca.0.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <4 x i16>, <4 x i16> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int16x4x2_t undef, <4 x i16> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x2_t %.fca.0.0.insert, <4 x i16> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int16x4x2_t %.fca.0.1.insert
}

define %struct.int32x2x2_t @test_vld2_lane_s32(i32* %a, [2 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_s32
; CHECK: ld2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i32>] %b.coerce, 1
  %0 = bitcast i32* %a to i8*
  %vld2_lane = tail call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, i32 1, i32 4)
  %vld2_lane.fca.0.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <2 x i32>, <2 x i32> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int32x2x2_t undef, <2 x i32> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x2_t %.fca.0.0.insert, <2 x i32> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int32x2x2_t %.fca.0.1.insert
}

define %struct.int64x1x2_t @test_vld2_lane_s64(i64* %a, [2 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_s64
; CHECK: ld2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x i64>] %b.coerce, 1
  %0 = bitcast i64* %a to i8*
  %vld2_lane = tail call { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, i32 0, i32 8)
  %vld2_lane.fca.0.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <1 x i64>, <1 x i64> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.int64x1x2_t undef, <1 x i64> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x2_t %.fca.0.0.insert, <1 x i64> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.int64x1x2_t %.fca.0.1.insert
}

define %struct.float32x2x2_t @test_vld2_lane_f32(float* %a, [2 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_f32
; CHECK: ld2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x float>] %b.coerce, 1
  %0 = bitcast float* %a to i8*
  %vld2_lane = tail call { <2 x float>, <2 x float> } @llvm.arm.neon.vld2lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, i32 1, i32 4)
  %vld2_lane.fca.0.extract = extractvalue { <2 x float>, <2 x float> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <2 x float>, <2 x float> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.float32x2x2_t undef, <2 x float> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x2_t %.fca.0.0.insert, <2 x float> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.float32x2x2_t %.fca.0.1.insert
}

define %struct.float64x1x2_t @test_vld2_lane_f64(double* %a, [2 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vld2_lane_f64
; CHECK: ld2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x double>] %b.coerce, 1
  %0 = bitcast double* %a to i8*
  %vld2_lane = tail call { <1 x double>, <1 x double> } @llvm.arm.neon.vld2lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, i32 0, i32 8)
  %vld2_lane.fca.0.extract = extractvalue { <1 x double>, <1 x double> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <1 x double>, <1 x double> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.float64x1x2_t undef, <1 x double> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x2_t %.fca.0.0.insert, <1 x double> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.float64x1x2_t %.fca.0.1.insert
}

define %struct.int16x8x3_t @test_vld3q_lane_s16(i16* %a, [3 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld3q_lane_s16
; CHECK: ld3 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i16>] %b.coerce, 2
  %0 = bitcast i16* %a to i8*
  %vld3_lane = tail call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, i32 7, i32 2)
  %vld3_lane.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int16x8x3_t undef, <8 x i16> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x3_t %.fca.0.0.insert, <8 x i16> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x3_t %.fca.0.1.insert, <8 x i16> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int16x8x3_t %.fca.0.2.insert
}

define %struct.int32x4x3_t @test_vld3q_lane_s32(i32* %a, [3 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld3q_lane_s32
; CHECK: ld3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i32>] %b.coerce, 2
  %0 = bitcast i32* %a to i8*
  %vld3_lane = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, i32 3, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int32x4x3_t undef, <4 x i32> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x3_t %.fca.0.0.insert, <4 x i32> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x3_t %.fca.0.1.insert, <4 x i32> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int32x4x3_t %.fca.0.2.insert
}

define %struct.int64x2x3_t @test_vld3q_lane_s64(i64* %a, [3 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld3q_lane_s64
; CHECK: ld3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i64>] %b.coerce, 2
  %0 = bitcast i64* %a to i8*
  %vld3_lane = tail call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, i32 1, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int64x2x3_t undef, <2 x i64> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x3_t %.fca.0.0.insert, <2 x i64> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x3_t %.fca.0.1.insert, <2 x i64> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int64x2x3_t %.fca.0.2.insert
}

define %struct.float32x4x3_t @test_vld3q_lane_f32(float* %a, [3 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vld3q_lane_f32
; CHECK: ld3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x float>] %b.coerce, 2
  %0 = bitcast float* %a to i8*
  %vld3_lane = tail call { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, i32 3, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.float32x4x3_t undef, <4 x float> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x3_t %.fca.0.0.insert, <4 x float> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x3_t %.fca.0.1.insert, <4 x float> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.float32x4x3_t %.fca.0.2.insert
}

define %struct.float64x2x3_t @test_vld3q_lane_f64(double* %a, [3 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vld3q_lane_f64
; CHECK: ld3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x double>] %b.coerce, 2
  %0 = bitcast double* %a to i8*
  %vld3_lane = tail call { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, i32 1, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.float64x2x3_t undef, <2 x double> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x3_t %.fca.0.0.insert, <2 x double> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x3_t %.fca.0.1.insert, <2 x double> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.float64x2x3_t %.fca.0.2.insert
}

define %struct.int8x8x3_t @test_vld3_lane_s8(i8* %a, [3 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_s8
; CHECK: ld3 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i8>] %b.coerce, 2
  %vld3_lane = tail call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, i32 7, i32 1)
  %vld3_lane.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int8x8x3_t undef, <8 x i8> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x3_t %.fca.0.0.insert, <8 x i8> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x3_t %.fca.0.1.insert, <8 x i8> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int8x8x3_t %.fca.0.2.insert
}

define %struct.int16x4x3_t @test_vld3_lane_s16(i16* %a, [3 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_s16
; CHECK: ld3 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i16>] %b.coerce, 2
  %0 = bitcast i16* %a to i8*
  %vld3_lane = tail call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, i32 3, i32 2)
  %vld3_lane.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int16x4x3_t undef, <4 x i16> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x3_t %.fca.0.0.insert, <4 x i16> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x3_t %.fca.0.1.insert, <4 x i16> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int16x4x3_t %.fca.0.2.insert
}

define %struct.int32x2x3_t @test_vld3_lane_s32(i32* %a, [3 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_s32
; CHECK: ld3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i32>] %b.coerce, 2
  %0 = bitcast i32* %a to i8*
  %vld3_lane = tail call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, i32 1, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int32x2x3_t undef, <2 x i32> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x3_t %.fca.0.0.insert, <2 x i32> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x3_t %.fca.0.1.insert, <2 x i32> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int32x2x3_t %.fca.0.2.insert
}

define %struct.int64x1x3_t @test_vld3_lane_s64(i64* %a, [3 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_s64
; CHECK: ld3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x i64>] %b.coerce, 2
  %0 = bitcast i64* %a to i8*
  %vld3_lane = tail call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, i32 0, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.int64x1x3_t undef, <1 x i64> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x3_t %.fca.0.0.insert, <1 x i64> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x3_t %.fca.0.1.insert, <1 x i64> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.int64x1x3_t %.fca.0.2.insert
}

define %struct.float32x2x3_t @test_vld3_lane_f32(float* %a, [3 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_f32
; CHECK: ld3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x float>] %b.coerce, 2
  %0 = bitcast float* %a to i8*
  %vld3_lane = tail call { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, i32 1, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.float32x2x3_t undef, <2 x float> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x3_t %.fca.0.0.insert, <2 x float> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x3_t %.fca.0.1.insert, <2 x float> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.float32x2x3_t %.fca.0.2.insert
}

define %struct.float64x1x3_t @test_vld3_lane_f64(double* %a, [3 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vld3_lane_f64
; CHECK: ld3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x double>] %b.coerce, 2
  %0 = bitcast double* %a to i8*
  %vld3_lane = tail call { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, i32 0, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.float64x1x3_t undef, <1 x double> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x3_t %.fca.0.0.insert, <1 x double> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x3_t %.fca.0.1.insert, <1 x double> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.float64x1x3_t %.fca.0.2.insert
}

define %struct.int8x16x4_t @test_vld4q_lane_s8(i8* %a, [4 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_s8
; CHECK: ld4 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vld3_lane = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4lane.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, <16 x i8> %b.coerce.fca.3.extract, i32 15, i32 1)
  %vld3_lane.fca.0.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int8x16x4_t undef, <16 x i8> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x16x4_t %.fca.0.0.insert, <16 x i8> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x16x4_t %.fca.0.1.insert, <16 x i8> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x16x4_t %.fca.0.2.insert, <16 x i8> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int8x16x4_t %.fca.0.3.insert
}

define %struct.int16x8x4_t @test_vld4q_lane_s16(i16* %a, [4 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_s16
; CHECK: ld4 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i16>] %b.coerce, 3
  %0 = bitcast i16* %a to i8*
  %vld3_lane = tail call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, <8 x i16> %b.coerce.fca.3.extract, i32 7, i32 2)
  %vld3_lane.fca.0.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int16x8x4_t undef, <8 x i16> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x8x4_t %.fca.0.0.insert, <8 x i16> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x8x4_t %.fca.0.1.insert, <8 x i16> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x8x4_t %.fca.0.2.insert, <8 x i16> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int16x8x4_t %.fca.0.3.insert
}

define %struct.int32x4x4_t @test_vld4q_lane_s32(i32* %a, [4 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_s32
; CHECK: ld4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i32>] %b.coerce, 3
  %0 = bitcast i32* %a to i8*
  %vld3_lane = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, <4 x i32> %b.coerce.fca.3.extract, i32 3, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int32x4x4_t undef, <4 x i32> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x4x4_t %.fca.0.0.insert, <4 x i32> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x4x4_t %.fca.0.1.insert, <4 x i32> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x4x4_t %.fca.0.2.insert, <4 x i32> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int32x4x4_t %.fca.0.3.insert
}

define %struct.int64x2x4_t @test_vld4q_lane_s64(i64* %a, [4 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_s64
; CHECK: ld4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i64>] %b.coerce, 3
  %0 = bitcast i64* %a to i8*
  %vld3_lane = tail call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, <2 x i64> %b.coerce.fca.3.extract, i32 1, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int64x2x4_t undef, <2 x i64> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x2x4_t %.fca.0.0.insert, <2 x i64> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x2x4_t %.fca.0.1.insert, <2 x i64> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x2x4_t %.fca.0.2.insert, <2 x i64> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int64x2x4_t %.fca.0.3.insert
}

define %struct.float32x4x4_t @test_vld4q_lane_f32(float* %a, [4 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_f32
; CHECK: ld4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x float>] %b.coerce, 3
  %0 = bitcast float* %a to i8*
  %vld3_lane = tail call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, <4 x float> %b.coerce.fca.3.extract, i32 3, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <4 x float>, <4 x float>, <4 x float>, <4 x float> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.float32x4x4_t undef, <4 x float> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x4x4_t %.fca.0.0.insert, <4 x float> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x4x4_t %.fca.0.1.insert, <4 x float> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x4x4_t %.fca.0.2.insert, <4 x float> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.float32x4x4_t %.fca.0.3.insert
}

define %struct.float64x2x4_t @test_vld4q_lane_f64(double* %a, [4 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vld4q_lane_f64
; CHECK: ld4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x double>] %b.coerce, 3
  %0 = bitcast double* %a to i8*
  %vld3_lane = tail call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, <2 x double> %b.coerce.fca.3.extract, i32 1, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <2 x double>, <2 x double>, <2 x double>, <2 x double> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.float64x2x4_t undef, <2 x double> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x2x4_t %.fca.0.0.insert, <2 x double> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x2x4_t %.fca.0.1.insert, <2 x double> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x2x4_t %.fca.0.2.insert, <2 x double> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.float64x2x4_t %.fca.0.3.insert
}

define %struct.int8x8x4_t @test_vld4_lane_s8(i8* %a, [4 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_s8
; CHECK: ld4 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i8>] %b.coerce, 3
  %vld3_lane = tail call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, <8 x i8> %b.coerce.fca.3.extract, i32 7, i32 1)
  %vld3_lane.fca.0.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int8x8x4_t undef, <8 x i8> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int8x8x4_t %.fca.0.0.insert, <8 x i8> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int8x8x4_t %.fca.0.1.insert, <8 x i8> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int8x8x4_t %.fca.0.2.insert, <8 x i8> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int8x8x4_t %.fca.0.3.insert
}

define %struct.int16x4x4_t @test_vld4_lane_s16(i16* %a, [4 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_s16
; CHECK: ld4 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i16>] %b.coerce, 3
  %0 = bitcast i16* %a to i8*
  %vld3_lane = tail call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, <4 x i16> %b.coerce.fca.3.extract, i32 3, i32 2)
  %vld3_lane.fca.0.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int16x4x4_t undef, <4 x i16> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int16x4x4_t %.fca.0.0.insert, <4 x i16> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int16x4x4_t %.fca.0.1.insert, <4 x i16> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int16x4x4_t %.fca.0.2.insert, <4 x i16> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int16x4x4_t %.fca.0.3.insert
}

define %struct.int32x2x4_t @test_vld4_lane_s32(i32* %a, [4 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_s32
; CHECK: ld4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i32>] %b.coerce, 3
  %0 = bitcast i32* %a to i8*
  %vld3_lane = tail call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, <2 x i32> %b.coerce.fca.3.extract, i32 1, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int32x2x4_t undef, <2 x i32> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int32x2x4_t %.fca.0.0.insert, <2 x i32> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int32x2x4_t %.fca.0.1.insert, <2 x i32> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int32x2x4_t %.fca.0.2.insert, <2 x i32> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int32x2x4_t %.fca.0.3.insert
}

define %struct.int64x1x4_t @test_vld4_lane_s64(i64* %a, [4 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_s64
; CHECK: ld4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x i64>] %b.coerce, 3
  %0 = bitcast i64* %a to i8*
  %vld3_lane = tail call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, <1 x i64> %b.coerce.fca.3.extract, i32 0, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.int64x1x4_t undef, <1 x i64> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.int64x1x4_t %.fca.0.0.insert, <1 x i64> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.int64x1x4_t %.fca.0.1.insert, <1 x i64> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.int64x1x4_t %.fca.0.2.insert, <1 x i64> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.int64x1x4_t %.fca.0.3.insert
}

define %struct.float32x2x4_t @test_vld4_lane_f32(float* %a, [4 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_f32
; CHECK: ld4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x float>] %b.coerce, 3
  %0 = bitcast float* %a to i8*
  %vld3_lane = tail call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, <2 x float> %b.coerce.fca.3.extract, i32 1, i32 4)
  %vld3_lane.fca.0.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <2 x float>, <2 x float>, <2 x float>, <2 x float> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.float32x2x4_t undef, <2 x float> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float32x2x4_t %.fca.0.0.insert, <2 x float> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float32x2x4_t %.fca.0.1.insert, <2 x float> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float32x2x4_t %.fca.0.2.insert, <2 x float> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.float32x2x4_t %.fca.0.3.insert
}

define %struct.float64x1x4_t @test_vld4_lane_f64(double* %a, [4 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vld4_lane_f64
; CHECK: ld4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x double>] %b.coerce, 3
  %0 = bitcast double* %a to i8*
  %vld3_lane = tail call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, <1 x double> %b.coerce.fca.3.extract, i32 0, i32 8)
  %vld3_lane.fca.0.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 2
  %vld3_lane.fca.3.extract = extractvalue { <1 x double>, <1 x double>, <1 x double>, <1 x double> } %vld3_lane, 3
  %.fca.0.0.insert = insertvalue %struct.float64x1x4_t undef, <1 x double> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.float64x1x4_t %.fca.0.0.insert, <1 x double> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.float64x1x4_t %.fca.0.1.insert, <1 x double> %vld3_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.float64x1x4_t %.fca.0.2.insert, <1 x double> %vld3_lane.fca.3.extract, 0, 3
  ret %struct.float64x1x4_t %.fca.0.3.insert
}

define void @test_vst1q_lane_s8(i8* %a, <16 x i8> %b) {
; CHECK-LABEL: test_vst1q_lane_s8
; CHECK: st1 {{{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <16 x i8> %b, i32 15
  store i8 %0, i8* %a, align 1
  ret void
}

define void @test_vst1q_lane_s16(i16* %a, <8 x i16> %b) {
; CHECK-LABEL: test_vst1q_lane_s16
; CHECK: st1 {{{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <8 x i16> %b, i32 7
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1q_lane_s32(i32* %a, <4 x i32> %b) {
; CHECK-LABEL: test_vst1q_lane_s32
; CHECK: st1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x i32> %b, i32 3
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1q_lane_s64(i64* %a, <2 x i64> %b) {
; CHECK-LABEL: test_vst1q_lane_s64
; CHECK: st1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x i64> %b, i32 1
  store i64 %0, i64* %a, align 8
  ret void
}

define void @test_vst1q_lane_f32(float* %a, <4 x float> %b) {
; CHECK-LABEL: test_vst1q_lane_f32
; CHECK: st1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x float> %b, i32 3
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1q_lane_f64(double* %a, <2 x double> %b) {
; CHECK-LABEL: test_vst1q_lane_f64
; CHECK: st1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x double> %b, i32 1
  store double %0, double* %a, align 8
  ret void
}

define void @test_vst1_lane_s8(i8* %a, <8 x i8> %b) {
; CHECK-LABEL: test_vst1_lane_s8
; CHECK: st1 {{{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <8 x i8> %b, i32 7
  store i8 %0, i8* %a, align 1
  ret void
}

define void @test_vst1_lane_s16(i16* %a, <4 x i16> %b) {
; CHECK-LABEL: test_vst1_lane_s16
; CHECK: st1 {{{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <4 x i16> %b, i32 3
  store i16 %0, i16* %a, align 2
  ret void
}

define void @test_vst1_lane_s32(i32* %a, <2 x i32> %b) {
; CHECK-LABEL: test_vst1_lane_s32
; CHECK: st1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x i32> %b, i32 1
  store i32 %0, i32* %a, align 4
  ret void
}

define void @test_vst1_lane_s64(i64* %a, <1 x i64> %b) {
; CHECK-LABEL: test_vst1_lane_s64
; CHECK: st1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <1 x i64> %b, i32 0
  store i64 %0, i64* %a, align 8
  ret void
}

define void @test_vst1_lane_f32(float* %a, <2 x float> %b) {
; CHECK-LABEL: test_vst1_lane_f32
; CHECK: st1 {{{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <2 x float> %b, i32 1
  store float %0, float* %a, align 4
  ret void
}

define void @test_vst1_lane_f64(double* %a, <1 x double> %b) {
; CHECK-LABEL: test_vst1_lane_f64
; CHECK: st1 {{{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %0 = extractelement <1 x double> %b, i32 0
  store double %0, double* %a, align 8
  ret void
}

define void @test_vst2q_lane_s8(i8* %a, [2 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_s8
; CHECK: st2 {{{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <16 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2lane.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, i32 15, i32 1)
  ret void
}

define void @test_vst2q_lane_s16(i16* %a, [2 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_s16
; CHECK: st2 {{{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i16>] %b.coerce, 1
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, i32 7, i32 2)
  ret void
}

define void @test_vst2q_lane_s32(i32* %a, [2 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_s32
; CHECK: st2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %b.coerce, 1
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, i32 3, i32 4)
  ret void
}

define void @test_vst2q_lane_s64(i64* %a, [2 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_s64
; CHECK: st2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i64>] %b.coerce, 1
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, i32 1, i32 8)
  ret void
}

define void @test_vst2q_lane_f32(float* %a, [2 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_f32
; CHECK: st2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x float>] %b.coerce, 1
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, i32 3, i32 4)
  ret void
}

define void @test_vst2q_lane_f64(double* %a, [2 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst2q_lane_f64
; CHECK: st2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x double>] %b.coerce, 1
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, i32 1, i32 8)
  ret void
}

define void @test_vst2_lane_s8(i8* %a, [2 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_s8
; CHECK: st2 {{{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <8 x i8>] %b.coerce, 1
  tail call void @llvm.arm.neon.vst2lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, i32 7, i32 1)
  ret void
}

define void @test_vst2_lane_s16(i16* %a, [2 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_s16
; CHECK: st2 {{{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <4 x i16>] %b.coerce, 1
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, i32 3, i32 2)
  ret void
}

define void @test_vst2_lane_s32(i32* %a, [2 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_s32
; CHECK: st2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x i32>] %b.coerce, 1
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, i32 1, i32 4)
  ret void
}

define void @test_vst2_lane_s64(i64* %a, [2 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_s64
; CHECK: st2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x i64>] %b.coerce, 1
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, i32 0, i32 8)
  ret void
}

define void @test_vst2_lane_f32(float* %a, [2 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_f32
; CHECK: st2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <2 x float>] %b.coerce, 1
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, i32 1, i32 4)
  ret void
}

define void @test_vst2_lane_f64(double* %a, [2 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst2_lane_f64
; CHECK: st2 {{{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [2 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [2 x <1 x double>] %b.coerce, 1
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst2lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, i32 0, i32 8)
  ret void
}

define void @test_vst3q_lane_s8(i8* %a, [3 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_s8
; CHECK: st3 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <16 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3lane.v16i8(i8* %a, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, i32 15, i32 1)
  ret void
}

define void @test_vst3q_lane_s16(i16* %a, [3 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_s16
; CHECK: st3 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i16>] %b.coerce, 2
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, i32 7, i32 2)
  ret void
}

define void @test_vst3q_lane_s32(i32* %a, [3 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_s32
; CHECK: st3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i32>] %b.coerce, 2
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, i32 3, i32 4)
  ret void
}

define void @test_vst3q_lane_s64(i64* %a, [3 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_s64
; CHECK: st3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i64>] %b.coerce, 2
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, i32 1, i32 8)
  ret void
}

define void @test_vst3q_lane_f32(float* %a, [3 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_f32
; CHECK: st3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x float>] %b.coerce, 2
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, i32 3, i32 4)
  ret void
}

define void @test_vst3q_lane_f64(double* %a, [3 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst3q_lane_f64
; CHECK: st3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x double>] %b.coerce, 2
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, i32 1, i32 8)
  ret void
}

define void @test_vst3_lane_s8(i8* %a, [3 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_s8
; CHECK: st3 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <8 x i8>] %b.coerce, 2
  tail call void @llvm.arm.neon.vst3lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, i32 7, i32 1)
  ret void
}

define void @test_vst3_lane_s16(i16* %a, [3 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_s16
; CHECK: st3 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <4 x i16>] %b.coerce, 2
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, i32 3, i32 2)
  ret void
}

define void @test_vst3_lane_s32(i32* %a, [3 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_s32
; CHECK: st3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x i32>] %b.coerce, 2
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, i32 1, i32 4)
  ret void
}

define void @test_vst3_lane_s64(i64* %a, [3 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_s64
; CHECK: st3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x i64>] %b.coerce, 2
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, i32 0, i32 8)
  ret void
}

define void @test_vst3_lane_f32(float* %a, [3 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_f32
; CHECK: st3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <2 x float>] %b.coerce, 2
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, i32 1, i32 4)
  ret void
}

define void @test_vst3_lane_f64(double* %a, [3 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst3_lane_f64
; CHECK: st3 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [3 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [3 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [3 x <1 x double>] %b.coerce, 2
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst3lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, i32 0, i32 8)
  ret void
}

define void @test_vst4q_lane_s8(i16* %a, [4 x <16 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_s8
; CHECK: st4 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v16i8(i8* %0, <16 x i8> %b.coerce.fca.0.extract, <16 x i8> %b.coerce.fca.1.extract, <16 x i8> %b.coerce.fca.2.extract, <16 x i8> %b.coerce.fca.3.extract, i32 15, i32 2)
  ret void
}

define void @test_vst4q_lane_s16(i16* %a, [4 x <8 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_s16
; CHECK: st4 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i16>] %b.coerce, 3
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v8i16(i8* %0, <8 x i16> %b.coerce.fca.0.extract, <8 x i16> %b.coerce.fca.1.extract, <8 x i16> %b.coerce.fca.2.extract, <8 x i16> %b.coerce.fca.3.extract, i32 7, i32 2)
  ret void
}

define void @test_vst4q_lane_s32(i32* %a, [4 x <4 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_s32
; CHECK: st4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i32>] %b.coerce, 3
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v4i32(i8* %0, <4 x i32> %b.coerce.fca.0.extract, <4 x i32> %b.coerce.fca.1.extract, <4 x i32> %b.coerce.fca.2.extract, <4 x i32> %b.coerce.fca.3.extract, i32 3, i32 4)
  ret void
}

define void @test_vst4q_lane_s64(i64* %a, [4 x <2 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_s64
; CHECK: st4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i64>] %b.coerce, 3
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2i64(i8* %0, <2 x i64> %b.coerce.fca.0.extract, <2 x i64> %b.coerce.fca.1.extract, <2 x i64> %b.coerce.fca.2.extract, <2 x i64> %b.coerce.fca.3.extract, i32 1, i32 8)
  ret void
}

define void @test_vst4q_lane_f32(float* %a, [4 x <4 x float>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_f32
; CHECK: st4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x float>] %b.coerce, 3
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v4f32(i8* %0, <4 x float> %b.coerce.fca.0.extract, <4 x float> %b.coerce.fca.1.extract, <4 x float> %b.coerce.fca.2.extract, <4 x float> %b.coerce.fca.3.extract, i32 3, i32 4)
  ret void
}

define void @test_vst4q_lane_f64(double* %a, [4 x <2 x double>] %b.coerce) {
; CHECK-LABEL: test_vst4q_lane_f64
; CHECK: st4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x double>] %b.coerce, 3
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2f64(i8* %0, <2 x double> %b.coerce.fca.0.extract, <2 x double> %b.coerce.fca.1.extract, <2 x double> %b.coerce.fca.2.extract, <2 x double> %b.coerce.fca.3.extract, i32 1, i32 8)
  ret void
}

define void @test_vst4_lane_s8(i8* %a, [4 x <8 x i8>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_s8
; CHECK: st4 {{{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b, {{v[0-9]+}}.b}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <8 x i8>] %b.coerce, 3
  tail call void @llvm.arm.neon.vst4lane.v8i8(i8* %a, <8 x i8> %b.coerce.fca.0.extract, <8 x i8> %b.coerce.fca.1.extract, <8 x i8> %b.coerce.fca.2.extract, <8 x i8> %b.coerce.fca.3.extract, i32 7, i32 1)
  ret void
}

define void @test_vst4_lane_s16(i16* %a, [4 x <4 x i16>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_s16
; CHECK: st4 {{{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h, {{v[0-9]+}}.h}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <4 x i16>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <4 x i16>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <4 x i16>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <4 x i16>] %b.coerce, 3
  %0 = bitcast i16* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v4i16(i8* %0, <4 x i16> %b.coerce.fca.0.extract, <4 x i16> %b.coerce.fca.1.extract, <4 x i16> %b.coerce.fca.2.extract, <4 x i16> %b.coerce.fca.3.extract, i32 3, i32 2)
  ret void
}

define void @test_vst4_lane_s32(i32* %a, [4 x <2 x i32>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_s32
; CHECK: st4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x i32>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x i32>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x i32>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x i32>] %b.coerce, 3
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2i32(i8* %0, <2 x i32> %b.coerce.fca.0.extract, <2 x i32> %b.coerce.fca.1.extract, <2 x i32> %b.coerce.fca.2.extract, <2 x i32> %b.coerce.fca.3.extract, i32 1, i32 4)
  ret void
}

define void @test_vst4_lane_s64(i64* %a, [4 x <1 x i64>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_s64
; CHECK: st4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x i64>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x i64>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x i64>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x i64>] %b.coerce, 3
  %0 = bitcast i64* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v1i64(i8* %0, <1 x i64> %b.coerce.fca.0.extract, <1 x i64> %b.coerce.fca.1.extract, <1 x i64> %b.coerce.fca.2.extract, <1 x i64> %b.coerce.fca.3.extract, i32 0, i32 8)
  ret void
}

define void @test_vst4_lane_f32(float* %a, [4 x <2 x float>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_f32
; CHECK: st4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <2 x float>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <2 x float>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <2 x float>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <2 x float>] %b.coerce, 3
  %0 = bitcast float* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v2f32(i8* %0, <2 x float> %b.coerce.fca.0.extract, <2 x float> %b.coerce.fca.1.extract, <2 x float> %b.coerce.fca.2.extract, <2 x float> %b.coerce.fca.3.extract, i32 1, i32 4)
  ret void
}

define void @test_vst4_lane_f64(double* %a, [4 x <1 x double>] %b.coerce) {
; CHECK-LABEL: test_vst4_lane_f64
; CHECK: st4 {{{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d, {{v[0-9]+}}.d}[{{[0-9]+}}], [x0]
entry:
  %b.coerce.fca.0.extract = extractvalue [4 x <1 x double>] %b.coerce, 0
  %b.coerce.fca.1.extract = extractvalue [4 x <1 x double>] %b.coerce, 1
  %b.coerce.fca.2.extract = extractvalue [4 x <1 x double>] %b.coerce, 2
  %b.coerce.fca.3.extract = extractvalue [4 x <1 x double>] %b.coerce, 3
  %0 = bitcast double* %a to i8*
  tail call void @llvm.arm.neon.vst4lane.v1f64(i8* %0, <1 x double> %b.coerce.fca.0.extract, <1 x double> %b.coerce.fca.1.extract, <1 x double> %b.coerce.fca.2.extract, <1 x double> %b.coerce.fca.3.extract, i32 0, i32 8)
  ret void
}

declare { <16 x i8>, <16 x i8> } @llvm.arm.neon.vld2lane.v16i8(i8*, <16 x i8>, <16 x i8>, i32, i32)
declare { <8 x i16>, <8 x i16> } @llvm.arm.neon.vld2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32, i32)
declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32)
declare { <2 x i64>, <2 x i64> } @llvm.arm.neon.vld2lane.v2i64(i8*, <2 x i64>, <2 x i64>, i32, i32)
declare { <4 x float>, <4 x float> } @llvm.arm.neon.vld2lane.v4f32(i8*, <4 x float>, <4 x float>, i32, i32)
declare { <2 x double>, <2 x double> } @llvm.arm.neon.vld2lane.v2f64(i8*, <2 x double>, <2 x double>, i32, i32)
declare { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32)
declare { <4 x i16>, <4 x i16> } @llvm.arm.neon.vld2lane.v4i16(i8*, <4 x i16>, <4 x i16>, i32, i32)
declare { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32, i32)
declare { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2.v1i64(i8*, i32)
declare { <2 x float>, <2 x float> } @llvm.arm.neon.vld2lane.v2f32(i8*, <2 x float>, <2 x float>, i32, i32)
declare { <1 x double>, <1 x double> } @llvm.arm.neon.vld2.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld3lane.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld3lane.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, i32, i32)
declare { <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld3lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32)
declare { <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld3lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, i32, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3.v1f64(i8*, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.arm.neon.vld4lane.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32, i32)
declare { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.arm.neon.vld4lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
declare { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.arm.neon.vld4lane.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i32, i32)
declare { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.arm.neon.vld4lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32)
declare { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.arm.neon.vld4lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32, i32)
declare { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.arm.neon.vld4lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32)
declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.arm.neon.vld4lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8*, i32)
declare { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.arm.neon.vld4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4.v1f64(i8*, i32)
declare { <1 x i64>, <1 x i64> } @llvm.arm.neon.vld2lane.v1i64(i8*, <1 x i64>, <1 x i64>, i32, i32)
declare { <1 x double>, <1 x double> } @llvm.arm.neon.vld2lane.v1f64(i8*, <1 x double>, <1 x double>, i32, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)
declare { <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld3lane.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, i32, i32)
declare { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)
declare { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.arm.neon.vld4lane.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, <1 x double>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v16i8(i8*, <16 x i8>, <16 x i8>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v2i64(i8*, <2 x i64>, <2 x i64>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v4f32(i8*, <4 x float>, <4 x float>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v2f64(i8*, <2 x double>, <2 x double>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v4i16(i8*, <4 x i16>, <4 x i16>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v1i64(i8*, <1 x i64>, <1 x i64>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v2f32(i8*, <2 x float>, <2 x float>, i32, i32)
declare void @llvm.arm.neon.vst2lane.v1f64(i8*, <1 x double>, <1 x double>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare void @llvm.arm.neon.vst3lane.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2i64(i8*, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2f64(i8*, <2 x double>, <2 x double>, <2 x double>, <2 x double>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32)
declare void @llvm.arm.neon.vst4lane.v1f64(i8*, <1 x double>, <1 x double>, <1 x double>, <1 x double>, i32, i32)