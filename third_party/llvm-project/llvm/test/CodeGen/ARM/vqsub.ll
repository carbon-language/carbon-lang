; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vqsubs8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vqsubs8:
;CHECK: vqsub.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.ssub.sat.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqsubs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vqsubs16:
;CHECK: vqsub.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqsubs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vqsubs32:
;CHECK: vqsub.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqsubs64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vqsubs64:
;CHECK: vqsub.s64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.ssub.sat.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqsubu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vqsubu8:
;CHECK: vqsub.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.usub.sat.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqsubu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vqsubu16:
;CHECK: vqsub.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.usub.sat.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqsubu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vqsubu32:
;CHECK: vqsub.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.usub.sat.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqsubu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vqsubu64:
;CHECK: vqsub.u64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = call <1 x i64> @llvm.usub.sat.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqsubQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vqsubQs8:
;CHECK: vqsub.s8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.ssub.sat.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqsubQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vqsubQs16:
;CHECK: vqsub.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqsubQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vqsubQs32:
;CHECK: vqsub.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqsubQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vqsubQs64:
;CHECK: vqsub.s64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqsubQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vqsubQu8:
;CHECK: vqsub.u8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqsubQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vqsubQu16:
;CHECK: vqsub.u16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqsubQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vqsubQu32:
;CHECK: vqsub.u32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.usub.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqsubQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vqsubQu64:
;CHECK: vqsub.u64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = call <2 x i64> @llvm.usub.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.ssub.sat.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.ssub.sat.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.ssub.sat.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.ssub.sat.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.usub.sat.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.usub.sat.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.usub.sat.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.usub.sat.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.ssub.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.ssub.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.usub.sat.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.usub.sat.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.usub.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.usub.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
