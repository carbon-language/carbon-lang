; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vmlai8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
;CHECK: vmlai8:
;CHECK: vmla.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

define <4 x i16> @vmlai16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlai16:
;CHECK: vmla.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

define <2 x i32> @vmlai32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlai32:
;CHECK: vmla.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

define <2 x float> @vmlaf32(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
;CHECK: vmlaf32:
;CHECK: vmla.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
	%tmp4 = mul <2 x float> %tmp2, %tmp3
	%tmp5 = add <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

define <16 x i8> @vmlaQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
;CHECK: vmlaQi8:
;CHECK: vmla.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

define <8 x i16> @vmlaQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
;CHECK: vmlaQi16:
;CHECK: vmla.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vmlaQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
;CHECK: vmlaQi32:
;CHECK: vmla.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x float> @vmlaQf32(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
;CHECK: vmlaQf32:
;CHECK: vmla.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
	%tmp4 = mul <4 x float> %tmp2, %tmp3
	%tmp5 = add <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}

define <8 x i16> @vmlals8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: vmlals8:
;CHECK: vmlal.s8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vmlals.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i16> %tmp4
}

define <4 x i32> @vmlals16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlals16:
;CHECK: vmlal.s16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vmlals.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

define <2 x i64> @vmlals32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlals32:
;CHECK: vmlal.s32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = call <2 x i64> @llvm.arm.neon.vmlals.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

define <8 x i16> @vmlalu8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: vmlalu8:
;CHECK: vmlal.u8
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = call <8 x i16> @llvm.arm.neon.vmlalu.v8i16(<8 x i16> %tmp1, <8 x i8> %tmp2, <8 x i8> %tmp3)
	ret <8 x i16> %tmp4
}

define <4 x i32> @vmlalu16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: vmlalu16:
;CHECK: vmlal.u16
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vmlalu.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

define <2 x i64> @vmlalu32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: vmlalu32:
;CHECK: vmlal.u32
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = call <2 x i64> @llvm.arm.neon.vmlalu.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

define arm_aapcs_vfpcc <4 x i32> @test_vmlal_lanes16(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %arg2_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmlal_lanes16
; CHECK: vmlal.s16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmlals.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmlal_lanes32(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %arg2_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmlal_lanes32
; CHECK: vmlal.s32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmlals.v2i64(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmlal_laneu16(<4 x i32> %arg0_uint32x4_t, <4 x i16> %arg1_uint16x4_t, <4 x i16> %arg2_uint16x4_t) nounwind readnone {
entry:
; CHECK: test_vmlal_laneu16
; CHECK: vmlal.u16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_uint16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmlalu.v4i32(<4 x i32> %arg0_uint32x4_t, <4 x i16> %arg1_uint16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmlal_laneu32(<2 x i64> %arg0_uint64x2_t, <2 x i32> %arg1_uint32x2_t, <2 x i32> %arg2_uint32x2_t) nounwind readnone {
entry:
; CHECK: test_vmlal_laneu32
; CHECK: vmlal.u32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_uint32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmlalu.v2i64(<2 x i64> %arg0_uint64x2_t, <2 x i32> %arg1_uint32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

declare <8 x i16> @llvm.arm.neon.vmlals.v8i16(<8 x i16>, <8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmlals.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmlals.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmlalu.v8i16(<8 x i16>, <8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmlalu.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmlalu.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone
