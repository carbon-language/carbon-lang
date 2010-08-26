; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @vld1i8(i8* %A) nounwind {
;CHECK: vld1i8:
;CHECK: vld1.8
	%tmp1 = call <8 x i8> @llvm.arm.neon.vld1.v8i8(i8* %A)
	ret <8 x i8> %tmp1
}

define <4 x i16> @vld1i16(i16* %A) nounwind {
;CHECK: vld1i16:
;CHECK: vld1.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call <4 x i16> @llvm.arm.neon.vld1.v4i16(i8* %tmp0)
	ret <4 x i16> %tmp1
}

define <2 x i32> @vld1i32(i32* %A) nounwind {
;CHECK: vld1i32:
;CHECK: vld1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call <2 x i32> @llvm.arm.neon.vld1.v2i32(i8* %tmp0)
	ret <2 x i32> %tmp1
}

define <2 x float> @vld1f(float* %A) nounwind {
;CHECK: vld1f:
;CHECK: vld1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call <2 x float> @llvm.arm.neon.vld1.v2f32(i8* %tmp0)
	ret <2 x float> %tmp1
}

define <1 x i64> @vld1i64(i64* %A) nounwind {
;CHECK: vld1i64:
;CHECK: vld1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call <1 x i64> @llvm.arm.neon.vld1.v1i64(i8* %tmp0)
	ret <1 x i64> %tmp1
}

define <16 x i8> @vld1Qi8(i8* %A) nounwind {
;CHECK: vld1Qi8:
;CHECK: vld1.8
	%tmp1 = call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %A)
	ret <16 x i8> %tmp1
}

define <8 x i16> @vld1Qi16(i16* %A) nounwind {
;CHECK: vld1Qi16:
;CHECK: vld1.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = call <8 x i16> @llvm.arm.neon.vld1.v8i16(i8* %tmp0)
	ret <8 x i16> %tmp1
}

define <4 x i32> @vld1Qi32(i32* %A) nounwind {
;CHECK: vld1Qi32:
;CHECK: vld1.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = call <4 x i32> @llvm.arm.neon.vld1.v4i32(i8* %tmp0)
	ret <4 x i32> %tmp1
}

define <4 x float> @vld1Qf(float* %A) nounwind {
;CHECK: vld1Qf:
;CHECK: vld1.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %tmp0)
	ret <4 x float> %tmp1
}

define <2 x i64> @vld1Qi64(i64* %A) nounwind {
;CHECK: vld1Qi64:
;CHECK: vld1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = call <2 x i64> @llvm.arm.neon.vld1.v2i64(i8* %tmp0)
	ret <2 x i64> %tmp1
}

declare <8 x i8>  @llvm.arm.neon.vld1.v8i8(i8*) nounwind readonly
declare <4 x i16> @llvm.arm.neon.vld1.v4i16(i8*) nounwind readonly
declare <2 x i32> @llvm.arm.neon.vld1.v2i32(i8*) nounwind readonly
declare <2 x float> @llvm.arm.neon.vld1.v2f32(i8*) nounwind readonly
declare <1 x i64> @llvm.arm.neon.vld1.v1i64(i8*) nounwind readonly

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8*) nounwind readonly
declare <8 x i16> @llvm.arm.neon.vld1.v8i16(i8*) nounwind readonly
declare <4 x i32> @llvm.arm.neon.vld1.v4i32(i8*) nounwind readonly
declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*) nounwind readonly
declare <2 x i64> @llvm.arm.neon.vld1.v2i64(i8*) nounwind readonly

; Radar 8355607
; Do not crash if the vld1 result is not used.
define void @unused_vld1_result() {
entry:
;CHECK: unused_vld1_result
;CHECK: vld1.32
  %0 = call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* undef) 
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() nounwind

