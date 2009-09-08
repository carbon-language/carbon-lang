; RUN: llc -mattr=+neon < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define arm_aapcs_vfpcc <8 x i16> @test_vqRdmulhQ_lanes16(<8 x i16> %arg0_int16x8_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulhQ_lanes16
; CHECK: vqrdmulh.s16 q0, q0, d2[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1> ; <<8 x i16>> [#uses=1]
  %1 = tail call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %arg0_int16x8_t, <8 x i16> %0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %1
}

declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone

define arm_aapcs_vfpcc <4 x i32> @test_vqRdmulhQ_lanes32(<4 x i32> %arg0_int32x4_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulhQ_lanes32
; CHECK: vqrdmulh.s32 q0, q0, d2[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i32>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i32> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define arm_aapcs_vfpcc <4 x i16> @test_vqRdmulh_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulh_lanes16
; CHECK: vqrdmulh.s16 d0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i16>> [#uses=1]
  ret <4 x i16> %1
}

declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>) nounwind readnone

define arm_aapcs_vfpcc <2 x i32> @test_vqRdmulh_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulh_lanes32
; CHECK: vqrdmulh.s32 d0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i32>> [#uses=1]
  ret <2 x i32> %1
}

declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
