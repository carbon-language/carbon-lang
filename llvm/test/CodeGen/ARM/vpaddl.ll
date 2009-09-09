; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vpaddl\\.s8} %t | count 2
; RUN: grep {vpaddl\\.s16} %t | count 2
; RUN: grep {vpaddl\\.s32} %t | count 2
; RUN: grep {vpaddl\\.u8} %t | count 2
; RUN: grep {vpaddl\\.u16} %t | count 2
; RUN: grep {vpaddl\\.u32} %t | count 2

define <4 x i16> @vpaddls8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vpaddls16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vpaddls32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

define <4 x i16> @vpaddlu8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vpaddlu16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vpaddlu32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

define <8 x i16> @vpaddlQs8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vpaddlQs16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vpaddlQs32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

define <8 x i16> @vpaddlQu8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vpaddlQu16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vpaddlQu32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

declare <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) nounwind readnone
