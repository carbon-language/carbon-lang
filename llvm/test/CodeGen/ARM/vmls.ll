; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vmlsi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
;CHECK: vmlsi8:
;CHECK: vmls.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = sub <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vmlsi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlsi16:
;CHECK: vmls.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = sub <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vmlsi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlsi32:
;CHECK: vmls.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = sub <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vmlsf32(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
;CHECK: vmlsf32:
;CHECK: vmls.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
	%tmp4 = fmul <2 x float> %tmp2, %tmp3
	%tmp5 = fsub <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

define <16 x i8> @vmlsQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
;CHECK: vmlsQi8:
;CHECK: vmls.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = sub <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vmlsQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
;CHECK: vmlsQi16:
;CHECK: vmls.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = sub <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vmlsQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
;CHECK: vmlsQi32:
;CHECK: vmls.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = sub <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vmlsQf32(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
;CHECK: vmlsQf32:
;CHECK: vmls.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
	%tmp4 = fmul <4 x float> %tmp2, %tmp3
	%tmp5 = fsub <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}

define <8 x i16> @vmlsls8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: vmlsls8:
;CHECK: vmlsl.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vmlsls.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i16> %tmp4
}

define <4 x i32> @vmlsls16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlsls16:
;CHECK: vmlsl.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vmlsls.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

define <2 x i64> @vmlsls32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlsls32:
;CHECK: vmlsl.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = call <2 x i64> @llvm.arm.neon.vmlsls.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

define <8 x i16> @vmlslu8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: vmlslu8:
;CHECK: vmlsl.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vmlslu.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i16> %tmp4
}

define <4 x i32> @vmlslu16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlslu16:
;CHECK: vmlsl.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vmlslu.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

define <2 x i64> @vmlslu32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlslu32:
;CHECK: vmlsl.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = call <2 x i64> @llvm.arm.neon.vmlslu.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

define arm_aapcs_vfpcc <4 x i32> @test_vmlsl_lanes16(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %arg2_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmlsl_lanes16
; CHECK: vmlsl.s16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmlsls.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmlsl_lanes32(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %arg2_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmlsl_lanes32
; CHECK: vmlsl.s32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmlsls.v2i64(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmlsl_laneu16(<4 x i32> %arg0_uint32x4_t, <4 x i16> %arg1_uint16x4_t, <4 x i16> %arg2_uint16x4_t) nounwind readnone {
entry:
; CHECK: test_vmlsl_laneu16
; CHECK: vmlsl.u16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_uint16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmlslu.v4i32(<4 x i32> %arg0_uint32x4_t, <4 x i16> %arg1_uint16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmlsl_laneu32(<2 x i64> %arg0_uint64x2_t, <2 x i32> %arg1_uint32x2_t, <2 x i32> %arg2_uint32x2_t) nounwind readnone {
entry:
; CHECK: test_vmlsl_laneu32
; CHECK: vmlsl.u32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_uint32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmlslu.v2i64(<2 x i64> %arg0_uint64x2_t, <2 x i32> %arg1_uint32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

declare <8 x i16> @llvm.arm.neon.vmlsls.v8i16(<8 x i16>, <8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmlsls.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmlsls.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmlslu.v8i16(<8 x i16>, <8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmlslu.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmlslu.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone
