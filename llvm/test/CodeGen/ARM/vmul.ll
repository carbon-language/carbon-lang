; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vmuli8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmuli8:
;CHECK: vmul.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = mul <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmuli16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmuli16:
;CHECK: vmul.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = mul <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmuli32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmuli32:
;CHECK: vmul.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = mul <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x float> @vmulf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vmulf32:
;CHECK: vmul.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fmul <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <8 x i8> @vmulp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulp8:
;CHECK: vmul.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @vmulQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQi8:
;CHECK: vmul.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = mul <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vmulQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vmulQi16:
;CHECK: vmul.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = mul <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vmulQi32:
;CHECK: vmul.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = mul <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x float> @vmulQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vmulQf32:
;CHECK: vmul.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fmul <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <16 x i8> @vmulQp8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQp8:
;CHECK: vmul.p8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8>  @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

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

define <8 x i16> @vmulls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulls8:
;CHECK: vmull.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmulls16:
;CHECK: vmull.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmulls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmulls32:
;CHECK: vmull.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullu8:
;CHECK: vmull.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmullu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmullu16:
;CHECK: vmull.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmullu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmullu32:
;CHECK: vmull.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullp8:
;CHECK: vmull.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullp.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes16
; CHECK: vmull.s16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes32
; CHECK: vmull.s32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_laneu16(<4 x i16> %arg0_uint16x4_t, <4 x i16> %arg1_uint16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu16
; CHECK: vmull.u16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_uint16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %arg0_uint16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_laneu32(<2 x i32> %arg0_uint32x2_t, <2 x i32> %arg1_uint32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu32
; CHECK: vmull.u32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_uint32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %arg0_uint32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

declare <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16>  @llvm.arm.neon.vmullp.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
