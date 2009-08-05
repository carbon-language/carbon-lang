; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vld1i8(i8* %A) nounwind {
;CHECK: vld1i8:
;CHECK: vld1.8
	%tmp1 = call <8 x i8> @llvm.arm.neon.vld1i.v8i8(i8* %A)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vld1i16(i16* %A) nounwind {
;CHECK: vld1i16:
;CHECK: vld1.16
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1i.v4i16(i16* %A)
	ret <4 x i16> %tmp1
}

define <2 x i32> @vld1i32(i32* %A) nounwind {
;CHECK: vld1i32:
;CHECK: vld1.32
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1i.v2i32(i32* %A)
	ret <2 x i32> %tmp1
}

define <2 x float> @vld1f(float* %A) nounwind {
;CHECK: vld1f:
;CHECK: vld1.32
	%tmp1 = call <2 x float> @llvm.arm.neon.vld1f.v2f32(float* %A)
	ret <2 x float> %tmp1
}

define <1 x i64> @vld1i64(i64* %A) nounwind {
;CHECK: vld1i64:
;CHECK: vld1.64
	%tmp1 = call <1 x i64> @llvm.arm.neon.vld1i.v1i64(i64* %A)
	ret <1 x i64> %tmp1
}

define <16 x i8> @vld1Qi8(i8* %A) nounwind {
;CHECK: vld1Qi8:
;CHECK: vld1.8
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1i.v16i8(i8* %A)
	ret <16 x i8> %tmp1
}

define <8 x i16> @vld1Qi16(i16* %A) nounwind {
;CHECK: vld1Qi16:
;CHECK: vld1.16
	%tmp1 = call <8 x i16> @llvm.arm.neon.vld1i.v8i16(i16* %A)
	ret <8 x i16> %tmp1
}

define <4 x i32> @vld1Qi32(i32* %A) nounwind {
;CHECK: vld1Qi32:
;CHECK: vld1.32
	%tmp1 = call <4 x i32> @llvm.arm.neon.vld1i.v4i32(i32* %A)
	ret <4 x i32> %tmp1
}

define <4 x float> @vld1Qf(float* %A) nounwind {
;CHECK: vld1Qf:
;CHECK: vld1.32
	%tmp1 = call <4 x float> @llvm.arm.neon.vld1f.v4f32(float* %A)
	ret <4 x float> %tmp1
}

define <2 x i64> @vld1Qi64(i64* %A) nounwind {
;CHECK: vld1Qi64:
;CHECK: vld1.64
	%tmp1 = call <2 x i64> @llvm.arm.neon.vld1i.v2i64(i64* %A)
	ret <2 x i64> %tmp1
}

declare <8 x i8>  @llvm.arm.neon.vld1i.v8i8(i8*) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vld1i.v4i16(i16*) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vld1i.v2i32(i32*) nounwind readnone
declare <2 x float> @llvm.arm.neon.vld1f.v2f32(float*) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vld1i.v1i64(i64*) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vld1i.v16i8(i8*) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vld1i.v8i16(i16*) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vld1i.v4i32(i32*) nounwind readnone
declare <4 x float> @llvm.arm.neon.vld1f.v4f32(float*) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vld1i.v2i64(i64*) nounwind readnone
