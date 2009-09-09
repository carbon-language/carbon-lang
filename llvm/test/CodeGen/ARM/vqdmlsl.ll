; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vqdmlsl\\.s16} %t | count 1
; RUN: grep {vqdmlsl\\.s32} %t | count 1

define <4 x i32> @vqdmlsls16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = call <4 x i32> @llvm.arm.neon.vqdmlsl.v4i32(<4 x i32> %tmp1, <4 x i16> %tmp2, <4 x i16> %tmp3)
	ret <4 x i32> %tmp4
}

define <2 x i64> @vqdmlsls32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = call <2 x i64> @llvm.arm.neon.vqdmlsl.v2i64(<2 x i64> %tmp1, <2 x i32> %tmp2, <2 x i32> %tmp3)
	ret <2 x i64> %tmp4
}

declare <4 x i32>  @llvm.arm.neon.vqdmlsl.v4i32(<4 x i32>, <4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmlsl.v2i64(<2 x i64>, <2 x i32>, <2 x i32>) nounwind readnone
