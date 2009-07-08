; RUN: llvm-as < %s | llc -march=arm -mattr=+neon > %t
; RUN: grep {vst1\\.8} %t | count 2
; RUN: grep {vst1\\.16} %t | count 2
; RUN: grep {vst1\\.32} %t | count 4
; RUN: grep {vst1\\.64} %t | count 2

define void @vst1i8(i8* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %B
	call void @llvm.arm.neon.vsti.v8i8(i8* %A, <8 x i8> %tmp1, i32 1)
	ret void
}

define void @vst1i16(i16* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %B
	call void @llvm.arm.neon.vsti.v4i16(i16* %A, <4 x i16> %tmp1, i32 1)
	ret void
}

define void @vst1i32(i32* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %B
	call void @llvm.arm.neon.vsti.v2i32(i32* %A, <2 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1f(float* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %B
	call void @llvm.arm.neon.vstf.v2f32(float* %A, <2 x float> %tmp1, i32 1)
	ret void
}

define void @vst1i64(i64* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %B
	call void @llvm.arm.neon.vsti.v1i64(i64* %A, <1 x i64> %tmp1, i32 1)
	ret void
}

define void @vst1Qi8(i8* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %B
	call void @llvm.arm.neon.vsti.v16i8(i8* %A, <16 x i8> %tmp1, i32 1)
	ret void
}

define void @vst1Qi16(i16* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %B
	call void @llvm.arm.neon.vsti.v8i16(i16* %A, <8 x i16> %tmp1, i32 1)
	ret void
}

define void @vst1Qi32(i32* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %B
	call void @llvm.arm.neon.vsti.v4i32(i32* %A, <4 x i32> %tmp1, i32 1)
	ret void
}

define void @vst1Qf(float* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %B
	call void @llvm.arm.neon.vstf.v4f32(float* %A, <4 x float> %tmp1, i32 1)
	ret void
}

define void @vst1Qi64(i64* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %B
	call void @llvm.arm.neon.vsti.v2i64(i64* %A, <2 x i64> %tmp1, i32 1)
	ret void
}

declare void @llvm.arm.neon.vsti.v8i8(i8*, <8 x i8>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v4i16(i16*, <4 x i16>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v2i32(i32*, <2 x i32>, i32) nounwind readnone
declare void @llvm.arm.neon.vstf.v2f32(float*, <2 x float>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v1i64(i64*, <1 x i64>, i32) nounwind readnone

declare void @llvm.arm.neon.vsti.v16i8(i8*, <16 x i8>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v8i16(i16*, <8 x i16>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v4i32(i32*, <4 x i32>, i32) nounwind readnone
declare void @llvm.arm.neon.vstf.v4f32(float*, <4 x float>, i32) nounwind readnone
declare void @llvm.arm.neon.vsti.v2i64(i64*, <2 x i64>, i32) nounwind readnone
