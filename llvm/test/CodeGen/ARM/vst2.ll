; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

define void @vst2i8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst2i8:
;CHECK: vst2.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst2.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1)
	ret void
}

define void @vst2i16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst2i16:
;CHECK: vst2.16
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst2.v4i16(i16* %A, <4 x i16> %tmp1, <4 x i16> %tmp1)
	ret void
}

define void @vst2i32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst2i32:
;CHECK: vst2.32
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst2.v2i32(i32* %A, <2 x i32> %tmp1, <2 x i32> %tmp1)
	ret void
}

define void @vst2f(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst2f:
;CHECK: vst2.32
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst2.v2f32(float* %A, <2 x float> %tmp1, <2 x float> %tmp1)
	ret void
}

declare void @llvm.arm.neon.vst2.v8i8(i8*, <8 x i8>, <8 x i8>) nounwind
declare void @llvm.arm.neon.vst2.v4i16(i8*, <4 x i16>, <4 x i16>) nounwind
declare void @llvm.arm.neon.vst2.v2i32(i8*, <2 x i32>, <2 x i32>) nounwind
declare void @llvm.arm.neon.vst2.v2f32(i8*, <2 x float>, <2 x float>) nounwind
