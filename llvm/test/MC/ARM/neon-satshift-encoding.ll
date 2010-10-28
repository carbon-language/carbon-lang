; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

define <8 x i8> @vqshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqshl.s8	d16, d16, d17   @ encoding: [0xb0,0x04,0x41,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqshl.s16	d16, d16, d17   @ encoding: [0xb0,0x04,0x51,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqshl.s32	d16, d16, d17   @ encoding: [0xb0,0x04,0x61,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqshl.s64	d16, d16, d17   @ encoding: [0xb0,0x04,0x71,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqshlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqshlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqshl.u8	d16, d16, d17   @ encoding: [0xb0,0x04,0x41,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqshlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqshl.u32	d16, d16, d17   @ encoding: [0xb0,0x04,0x61,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqshlu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqshl.u64	d16, d16, d17   @ encoding: [0xb0,0x04,0x71,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqshl.s8	q8, q8, q9      @ encoding: [0xf0,0x04,0x42,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqshl.s16	q8, q8, q9      @ encoding: [0xf0,0x04,0x52,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqshl.s32	q8, q8, q9      @ encoding: [0xf0,0x04,0x62,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqshlQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqshl.u8	q8, q8, q9      @ encoding: [0xf0,0x04,0x42,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqshlQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqshl.u16	q8, q8, q9      @ encoding: [0xf0,0x04,0x52,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqshlQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqshl.u32	q8, q8, q9      @ encoding: [0xf0,0x04,0x62,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqshlQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqshl.u64	q8, q8, q9      @ encoding: [0xf0,0x04,0x72,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i8> @vqshls_n8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vqshl.s8	d16, d16, #7    @ encoding: [0x30,0x07,0xcf,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshls_n16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vqshl.s16	d16, d16, #15   @ encoding: [0x30,0x07,0xdf,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshls_n32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vqshl.s32	d16, d16, #31   @ encoding: [0x30,0x07,0xff,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshls_n64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vqshl.s64	d16, d16, #63   @ encoding: [0xb0,0x07,0xff,0xf2]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vqshlu_n8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vqshl.u8	d16, d16, #7    @ encoding: [0x30,0x07,0xcf,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshlu_n16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vqshl.u16	d16, d16, #15   @ encoding: [0x30,0x07,0xdf,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshlu_n32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vqshl.u32	d16, d16, #31   @ encoding: [0x30,0x07,0xff,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshlu_n64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vqshl.u64	d16, d16, #63   @ encoding: [0xb0,0x07,0xff,0xf3]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <8 x i8> @vqshlsu_n8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vqshlu.s8	d16, d16, #7    @ encoding: [0x30,0x06,0xcf,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftsu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshlsu_n16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vqshlu.s16	d16, d16, #15   @ encoding: [0x30,0x06,0xdf,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftsu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshlsu_n32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vqshlu.s32	d16, d16, #31   @ encoding: [0x30,0x06,0xff,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftsu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i32> %tmp2
}

define <1 x i64> @vqshlsu_n64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vqshlu.s64	d16, d16, #63   @ encoding: [0xb0,0x06,0xff,0xf3]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vqshiftsu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 63 >)
	ret <1 x i64> %tmp2
}

define <16 x i8> @vqshlQs_n8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vqshl.s8	q8, q8, #7      @ encoding: [0x70,0x07,0xcf,0xf2]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQs_n16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshl.s16	q8, q8, #15     @ encoding: [0x70,0x07,0xdf,0xf2]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQs_n32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshl.s32	q8, q8, #31     @ encoding: [0x70,0x07,0xff,0xf2]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQs_n64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshl.s64	q8, q8, #63     @ encoding: [0xf0,0x07,0xff,0xf2]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vqshlQu_n8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vqshl.u8	q8, q8, #7      @ encoding: [0x70,0x07,0xcf,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQu_n16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshl.u16	q8, q8, #15     @ encoding: [0x70,0x07,0xdf,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQu_n32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshl.u32	q8, q8, #31     @ encoding: [0x70,0x07,0xff,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQu_n64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshl.u64	q8, q8, #63     @ encoding: [0xf0,0x07,0xff,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

define <16 x i8> @vqshlQsu_n8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vqshlu.s8	q8, q8, #7      @ encoding: [0x70,0x06,0xcf,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vqshiftsu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <16 x i8> %tmp2
}

define <8 x i16> @vqshlQsu_n16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshlu.s16	q8, q8, #15     @ encoding: [0x70,0x06,0xdf,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vqshiftsu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >)
	ret <8 x i16> %tmp2
}

define <4 x i32> @vqshlQsu_n32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshlu.s32	q8, q8, #31     @ encoding: [0x70,0x06,0xff,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vqshiftsu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 31, i32 31, i32 31, i32 31 >)
	ret <4 x i32> %tmp2
}

define <2 x i64> @vqshlQsu_n64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshlu.s64	q8, q8, #63     @ encoding: [0xf0,0x06,0xff,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vqshiftsu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 63, i64 63 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftsu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftsu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftsu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqshiftsu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqshiftsu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqshiftsu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqshiftsu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqshiftsu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vqrshls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK:   vqrshl.s8	d16, d16, d17   @ encoding: [0xb0,0x05,0x41,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqrshls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqrshl.s16	d16, d16, d17   @ encoding: [0xb0,0x05,0x51,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqrshls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqrshl.s32	d16, d16, d17   @ encoding: [0xb0,0x05,0x61,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqrshls64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqrshl.s64	d16, d16, d17   @ encoding: [0xb0,0x05,0x71,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <8 x i8> @vqrshlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqrshl.u8	d16, d16, d17   @ encoding: [0xb0,0x05,0x41,0xf3
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vqrshlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqrshl.u16	d16, d16, d17   @ encoding: [0xb0,0x05,0x51,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqrshlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqrshl.u32	d16, d16, d17   @ encoding: [0xb0,0x05,0x61,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <1 x i64> @vqrshlu64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqrshl.u64	d16, d16, d17   @ encoding: [0xb0,0x05,0x71,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

define <16 x i8> @vqrshlQs8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqrshl.s8	q8, q8, q9      @ encoding: [0xf0,0x05,0x42,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqrshlQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqrshl.s16	q8, q8, q9      @ encoding: [0xf0,0x05,0x52,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqrshlQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqrshl.s32	q8, q8, q9      @ encoding: [0xf0,0x05,0x62,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqrshlQs64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqrshl.s64	q8, q8, q9      @ encoding: [0xf0,0x05,0x72,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

define <16 x i8> @vqrshlQu8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqrshl.u8	q8, q8, q9      @ encoding: [0xf0,0x05,0x42,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

define <8 x i16> @vqrshlQu16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqrshl.u16	q8, q8, q9      @ encoding: [0xf0,0x05,0x52,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqrshlQu32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqrshl.u32	q8, q8, q9      @ encoding: [0xf0,0x05,0x62,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqrshlQu64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqrshl.u64	q8, q8, q9      @ encoding: [0xf0,0x05,0x72,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

declare <16 x i8> @llvm.arm.neon.vqrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vqshrns8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshrn.s16	d16, q8, #8     @ encoding: [0x30,0x09,0xc8,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftns.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshrns16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshrn.s32	d16, q8, #16    @ encoding: [0x30,0x09,0xd0,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftns.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshrns32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshrn.s64	d16, q8, #32    @ encoding: [0x30,0x09,0xe0,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftns.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqshrnu8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshrn.u16	d16, q8, #8     @ encoding: [0x30,0x09,0xc8,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftnu.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshrnu16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshrn.u32	d16, q8, #16    @ encoding: [0x30,0x09,0xd0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftnu.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshrnu32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshrn.u64	d16, q8, #32    @ encoding: [0x30,0x09,0xe0,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftnu.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqshruns8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqshrun.s16	d16, q8, #8     @ encoding: [0x30,0x08,0xc8,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqshiftnsu.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqshruns16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqshrun.s32	d16, q8, #16    @ encoding: [0x30,0x08,0xd0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqshiftnsu.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqshruns32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqshrun.s64	d16, q8, #32    @ encoding: [0x30,0x08,0xe0,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqshiftnsu.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqshiftns.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftns.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftns.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftnu.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftnu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftnu.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqshiftnsu.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqshiftnsu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqshiftnsu.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vqrshrns8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqrshrn.s16	d16, q8, #8     @ encoding: [0x70,0x09,0xc8,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqrshiftns.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqrshrns16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqrshrn.s32	d16, q8, #16    @ encoding: [0x70,0x09,0xd0,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqrshiftns.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqrshrns32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqrshrn.s64	d16, q8, #32    @ encoding: [0x70,0x09,0xe0,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqrshiftns.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqrshrnu8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqrshrn.u16	d16, q8, #8     @ encoding: [0x70,0x09,0xc8,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqrshiftnu.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqrshrnu16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqrshrn.u32	d16, q8, #16    @ encoding: [0x70,0x09,0xd0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqrshiftnu.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqrshrnu32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqrshrn.u64	d16, q8, #32    @ encoding: [0x70,0x09,0xe0,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqrshiftnu.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqrshruns8(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vqrshrun.s16	d16, q8, #8     @ encoding: [0x70,0x08,0xc8,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqrshiftnsu.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqrshruns16(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vqrshrun.s32	d16, q8, #16    @ encoding: [0x70,0x08,0xd0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqrshiftnsu.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqrshruns32(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vqrshrun.s64	d16, q8, #32    @ encoding: [0x70,0x08,0xe0,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqrshiftnsu.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqrshiftns.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshiftns.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshiftns.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqrshiftnu.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshiftnu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshiftnu.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqrshiftnsu.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqrshiftnsu.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrshiftnsu.v2i32(<2 x i64>, <2 x i64>) nounwind readnone
