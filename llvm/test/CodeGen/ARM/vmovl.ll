; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vmovl\\.s8} %t | count 1
; RUN: grep {vmovl\\.s16} %t | count 1
; RUN: grep {vmovl\\.s32} %t | count 1
; RUN: grep {vmovl\\.u8} %t | count 1
; RUN: grep {vmovl\\.u16} %t | count 1
; RUN: grep {vmovl\\.u32} %t | count 1

define <8 x i16> @vmovls8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vmovls.v8i16(<8 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovls16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vmovls.v4i32(<4 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovls32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vmovls.v2i64(<2 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

define <8 x i16> @vmovlu8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vmovlu.v8i16(<8 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovlu16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vmovlu.v4i32(<4 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovlu32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vmovlu.v2i64(<2 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

declare <8 x i16> @llvm.arm.neon.vmovls.v8i16(<8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmovls.v4i32(<4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmovls.v2i64(<2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmovlu.v8i16(<8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmovlu.v4i32(<4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmovlu.v2i64(<2 x i32>) nounwind readnone
