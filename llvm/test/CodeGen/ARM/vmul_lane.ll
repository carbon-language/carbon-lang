; RUN: llc -mattr=+neon < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define arm_aapcs_vfpcc <2 x float> @test_vmul_lanef32(<2 x float> %arg0_float32x2_t, <2 x float> %arg1_float32x2_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanef32:
; CHECK: vmul.f32 d0, d0, d1[0]
  %0 = shufflevector <2 x float> %arg1_float32x2_t, <2 x float> undef, <2 x i32> zeroinitializer ; <<2 x float>> [#uses=1]
  %1 = fmul <2 x float> %0, %arg0_float32x2_t     ; <<2 x float>> [#uses=1]
  ret <2 x float> %1
}

define arm_aapcs_vfpcc <4 x i16> @test_vmul_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanes16:
; CHECK: vmul.i16 d0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses$
  %1 = mul <4 x i16> %0, %arg0_int16x4_t          ; <<4 x i16>> [#uses=1]
  ret <4 x i16> %1
}

define arm_aapcs_vfpcc <2 x i32> @test_vmul_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanes32:
; CHECK: vmul.i32 d0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = mul <2 x i32> %0, %arg0_int32x2_t          ; <<2 x i32>> [#uses=1]
  ret <2 x i32> %1
}

define arm_aapcs_vfpcc <4 x float> @test_vmulQ_lanef32(<4 x float> %arg0_float32x4_t, <2 x float> %arg1_float32x2_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanef32:
; CHECK: vmul.f32 q0, q0, d2[1]
  %0 = shufflevector <2 x float> %arg1_float32x2_t, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x float>$
  %1 = fmul <4 x float> %0, %arg0_float32x4_t     ; <<4 x float>> [#uses=1]
  ret <4 x float> %1
}

define arm_aapcs_vfpcc <8 x i16> @test_vmulQ_lanes16(<8 x i16> %arg0_int16x8_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanes16:
; CHECK: vmul.i16 q0, q0, d2[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %1 = mul <8 x i16> %0, %arg0_int16x8_t          ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmulQ_lanes32(<4 x i32> %arg0_int32x4_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanes32:
; CHECK: vmul.i32 q0, q0, d2[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i32>> [#uses$
  %1 = mul <4 x i32> %0, %arg0_int32x4_t          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}
