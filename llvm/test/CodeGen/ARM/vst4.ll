; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define void @vst4i8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst4i8:
;CHECK: vst4.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst4.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1)
	ret void
}

define void @vst4i16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst4i16:
;CHECK: vst4.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst4.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1)
	ret void
}

define void @vst4i32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst4i32:
;CHECK: vst4.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst4.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1)
	ret void
}

define void @vst4f(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst4f:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst4.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst4i64(i64* %A, <1 x i64>* %B) nounwind {
;CHECK: vst4i64:
;CHECK: vst1.64
	%tmp0 = bitcast i64* %A to i8*
	%tmp1 = load <1 x i64>* %B
	call void @llvm.arm.neon.vst4.v1i64(i8* %tmp0, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, <1 x i64> %tmp1, i32 1)
	ret void
}

define void @vst4Qi8(i8* %A, <16 x i8>* %B) nounwind {
;CHECK: vst4Qi8:
;CHECK: vst4.8
;CHECK: vst4.8
	%tmp1 = load <16 x i8>* %B
	call void @llvm.arm.neon.vst4.v16i8(i8* %A, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, <16 x i8> %tmp1, i32 1)
	ret void
}

define void @vst4Qi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst4Qi16:
;CHECK: vst4.16
;CHECK: vst4.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst4.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1)
	ret void
}

define void @vst4Qi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst4Qi32:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst4.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst4Qf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst4Qf:
;CHECK: vst4.32
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst4.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst4.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32) nounwind
declare void @llvm.arm.neon.vst4.v1i64(i8*, <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64>, i32) nounwind

declare void @llvm.arm.neon.vst4.v16i8(i8*, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32) nounwind
declare void @llvm.arm.neon.vst4.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32) nounwind
declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind
declare void @llvm.arm.neon.vst4.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32) nounwind
