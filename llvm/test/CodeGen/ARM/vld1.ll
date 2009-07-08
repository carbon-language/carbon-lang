; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vld1\\.8} %t | count 2
; RUN: grep {vld1\\.16} %t | count 2
; RUN: grep {vld1\\.32} %t | count 4
; RUN: grep {vld1\\.64} %t | count 2

define <8 x i8> @vld1i8(i8* %A) nounwind {
	%tmp1 = call <8 x i8> @llvm.arm.neon.vldi.v8i8(i8* %A, i32 1)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vld1i16(i16* %A) nounwind {
	%tmp1 = call <4 x i16> @llvm.arm.neon.vldi.v4i16(i16* %A, i32 1)
	ret <4 x i16> %tmp1
}

define <2 x i32> @vld1i32(i32* %A) nounwind {
	%tmp1 = call <2 x i32> @llvm.arm.neon.vldi.v2i32(i32* %A, i32 1)
	ret <2 x i32> %tmp1
}

define <2 x float> @vld1f(float* %A) nounwind {
	%tmp1 = call <2 x float> @llvm.arm.neon.vldf.v2f32(float* %A, i32 1)
	ret <2 x float> %tmp1
}

define <1 x i64> @vld1i64(i64* %A) nounwind {
	%tmp1 = call <1 x i64> @llvm.arm.neon.vldi.v1i64(i64* %A, i32 1)
	ret <1 x i64> %tmp1
}

define <16 x i8> @vld1Qi8(i8* %A) nounwind {
	%tmp1 = call <16 x i8> @llvm.arm.neon.vldi.v16i8(i8* %A, i32 1)
	ret <16 x i8> %tmp1
}

define <8 x i16> @vld1Qi16(i16* %A) nounwind {
	%tmp1 = call <8 x i16> @llvm.arm.neon.vldi.v8i16(i16* %A, i32 1)
	ret <8 x i16> %tmp1
}

define <4 x i32> @vld1Qi32(i32* %A) nounwind {
	%tmp1 = call <4 x i32> @llvm.arm.neon.vldi.v4i32(i32* %A, i32 1)
	ret <4 x i32> %tmp1
}

define <4 x float> @vld1Qf(float* %A) nounwind {
	%tmp1 = call <4 x float> @llvm.arm.neon.vldf.v4f32(float* %A, i32 1)
	ret <4 x float> %tmp1
}

define <2 x i64> @vld1Qi64(i64* %A) nounwind {
	%tmp1 = call <2 x i64> @llvm.arm.neon.vldi.v2i64(i64* %A, i32 1)
	ret <2 x i64> %tmp1
}

declare <8 x i8>  @llvm.arm.neon.vldi.v8i8(i8*, i32) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vldi.v4i16(i16*, i32) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vldi.v2i32(i32*, i32) nounwind readnone
declare <2 x float> @llvm.arm.neon.vldf.v2f32(float*, i32) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vldi.v1i64(i64*, i32) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vldi.v16i8(i8*, i32) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vldi.v8i16(i16*, i32) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vldi.v4i32(i32*, i32) nounwind readnone
declare <4 x float> @llvm.arm.neon.vldf.v4f32(float*, i32) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vldi.v2i64(i64*, i32) nounwind readnone
