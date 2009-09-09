; RUN: llc < %s -march=arm -mattr=+neon > %t
; RUN: grep {vrshrn\\.i16} %t | count 1
; RUN: grep {vrshrn\\.i32} %t | count 1
; RUN: grep {vrshrn\\.i64} %t | count 1

define <8 x i8> @vrshrns8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshiftn.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vrshrns16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vrshrns32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vrshiftn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
