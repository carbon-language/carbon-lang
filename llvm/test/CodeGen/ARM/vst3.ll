; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define void @vst3i8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst3i8:
;CHECK: vst3.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst3.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1)
	ret void
}

define void @vst3i16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst3i16:
;CHECK: vst3.16
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst3.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1)
	ret void
}

define void @vst3i32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst3i32:
;CHECK: vst3.32
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst3.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1)
	ret void
}

define void @vst3f(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst3f:
;CHECK: vst3.32
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst3.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1)
	ret void
}

declare void @llvm.arm.neon.vst3.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>) nounwind
declare void @llvm.arm.neon.vst3.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>) nounwind
declare void @llvm.arm.neon.vst3.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>) nounwind
declare void @llvm.arm.neon.vst3.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>) nounwind
