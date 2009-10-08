; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <4 x i32> @vqdmulls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vqdmulls16:
;CHECK: vqdmull.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqdmulls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vqdmulls32:
;CHECK: vqdmull.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <4 x i32>  @llvm.arm.neon.vqdmull.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmull.v2i64(<2 x i32>, <2 x i32>) nounwind readnone
