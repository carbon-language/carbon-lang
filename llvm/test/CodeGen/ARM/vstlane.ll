; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define void @vst2lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst2lanei8:
;CHECK: vst2.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst2lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst2lanei16:
;CHECK: vst2.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst2lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst2lanei32:
;CHECK: vst2.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst2lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst2lanef:
;CHECK: vst2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst2lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst2laneQi16:
;CHECK: vst2.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst2lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst2laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst2laneQi32:
;CHECK: vst2.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst2lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 1)
	ret void
}

define void @vst2laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst2laneQf:
;CHECK: vst2.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst2lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, i32 3, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst2lane.v8i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4i16(i8*, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v2i32(i8*, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v2f32(i8*, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst2lane.v8i16(i8*, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst2lane.v4f32(i8*, <4 x float>, <4 x float>, i32, i32) nounwind

define void @vst3lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst3lanei8:
;CHECK: vst3.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst3lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst3lanei16:
;CHECK: vst3.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst3lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst3lanei32:
;CHECK: vst3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst3lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst3lanef:
;CHECK: vst3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst3lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst3laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst3laneQi16:
;CHECK: vst3.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst3lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 6, i32 1)
	ret void
}

define void @vst3laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst3laneQi32:
;CHECK: vst3.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst3lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 0, i32 1)
	ret void
}

define void @vst3laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst3laneQf:
;CHECK: vst3.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst3lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst3lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst3lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst3lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind


define void @vst4lanei8(i8* %A, <8 x i8>* %B) nounwind {
;CHECK: vst4lanei8:
;CHECK: vst4.8
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vst4lane.v8i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4lanei16(i16* %A, <4 x i16>* %B) nounwind {
;CHECK: vst4lanei16:
;CHECK: vst4.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vst4lane.v4i16(i8* %tmp0, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, <4 x i16> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4lanei32(i32* %A, <2 x i32>* %B) nounwind {
;CHECK: vst4lanei32:
;CHECK: vst4.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vst4lane.v2i32(i8* %tmp0, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, <2 x i32> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4lanef(float* %A, <2 x float>* %B) nounwind {
;CHECK: vst4lanef:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vst4lane.v2f32(i8* %tmp0, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, <2 x float> %tmp1, i32 1, i32 1)
	ret void
}

define void @vst4laneQi16(i16* %A, <8 x i16>* %B) nounwind {
;CHECK: vst4laneQi16:
;CHECK: vst4.16
	%tmp0 = bitcast i16* %A to i8*
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vst4lane.v8i16(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 7, i32 1)
	ret void
}

define void @vst4laneQi32(i32* %A, <4 x i32>* %B) nounwind {
;CHECK: vst4laneQi32:
;CHECK: vst4.32
	%tmp0 = bitcast i32* %A to i8*
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vst4lane.v4i32(i8* %tmp0, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, <4 x i32> %tmp1, i32 2, i32 1)
	ret void
}

define void @vst4laneQf(float* %A, <4 x float>* %B) nounwind {
;CHECK: vst4laneQf:
;CHECK: vst4.32
	%tmp0 = bitcast float* %A to i8*
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vst4lane.v4f32(i8* %tmp0, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, <4 x float> %tmp1, i32 1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vst4lane.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4i16(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v2i32(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v2f32(i8*, <2 x float>, <2 x float>, <2 x float>, <2 x float>, i32, i32) nounwind

declare void @llvm.arm.neon.vst4lane.v8i16(i8*, <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32) nounwind
declare void @llvm.arm.neon.vst4lane.v4f32(i8*, <4 x float>, <4 x float>, <4 x float>, <4 x float>, i32, i32) nounwind
