; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; CHECK: vshls_8xi8
define <8 x i8> @vshls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vshl.u8	d16, d17, d16           @ encoding: [0xa1,0x04,0x40,0xf3]
	%tmp3 = shl <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vshls_4xi16
define <4 x i16> @vshls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vshl.u16	d16, d17, d16   @ encoding: [0xa1,0x04,0x50,0xf3]
	%tmp3 = shl <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vshls_2xi32
define <2 x i32> @vshls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vshl.u32	d16, d17, d16   @ encoding: [0xa1,0x04,0x60,0xf3]
	%tmp3 = shl <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vshls_1xi64
define <1 x i64> @vshls_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vshl.u64	d16, d17, d16   @ encoding: [0xa1,0x04,0x70,0xf3]
	%tmp3 = shl <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

; CHECK: vshli_8xi8
define <8 x i8> @vshli_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshl.i8	d16, d16, #7            @ encoding: [0x30,0x05,0xcf,0xf2]
	%tmp2 = shl <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

; CHECK: vshli_4xi16
define <4 x i16> @vshli_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshl.i16	d16, d16, #15   @ encoding: [0x30,0x05,0xdf,0xf2
	%tmp2 = shl <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

; CHECK: vshli_2xi32
define <2 x i32> @vshli_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshl.i32	d16, d16, #31   @ encoding: [0x30,0x05,0xff,0xf2]
	%tmp2 = shl <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

; CHECK: vshli_1xi64
define <1 x i64> @vshli_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vshl.i64	d16, d16, #63   @ encoding: [0xb0,0x05,0xff,0xf2]
	%tmp2 = shl <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

; CHECK: vshls_16xi8
define <16 x i8> @vshls_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vshl.u8	q8, q8, q9              @ encoding: [0xe0,0x04,0x42,0xf3]
	%tmp3 = shl <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vshls_8xi16
define <8 x i16> @vshls_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shl <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

; CHECK: vshls_4xi32
define <4 x i32> @vshls_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vshl.u32	q8, q8, q9      @ encoding: [0xe0,0x04,0x62,0xf3]
	%tmp3 = shl <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

; CHECK: vshls_2xi64
define <2 x i64> @vshls_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vshl.u64	q8, q8, q9      @ encoding: [0xe0,0x04,0x72,0xf3]
	%tmp3 = shl <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

; CHECK: vshli_16xi8
define <16 x i8> @vshli_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vshl.i8	q8, q8, #7              @ encoding: [0x70,0x05,0xcf,0xf2]
	%tmp2 = shl <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

; CHECK: vshli_8xi16
define <8 x i16> @vshli_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vshl.i16	q8, q8, #15     @ encoding: [0x70,0x05,0xdf,0xf2]
	%tmp2 = shl <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

; CHECK: vshli_4xi32
define <4 x i32> @vshli_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vshl.i32	q8, q8, #31     @ encoding: [0x70,0x05,0xff,0xf2]
	%tmp2 = shl <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

; CHECK: vshli_2xi64
define <2 x i64> @vshli_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vshl.i64	q8, q8, #63     @ encoding: [0xf0,0x05,0xff,0xf2]
	%tmp2 = shl <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}

; CHECK: vshru_8xi8
define <8 x i8> @vshru_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshr.u8	d16, d16, #8            @ encoding: [0x30,0x00,0xc8,0xf3]
	%tmp2 = lshr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

; CHECK: vshru_4xi16
define <4 x i16> @vshru_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshr.u16	d16, d16, #16   @ encoding: [0x30,0x00,0xd0,0xf3]
	%tmp2 = lshr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

; CHECK: vshru_2xi32
define <2 x i32> @vshru_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshr.u32	d16, d16, #32   @ encoding: [0x30,0x00,0xe0,0xf3]
	%tmp2 = lshr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

; CHECK: vshru_1xi64
define <1 x i64> @vshru_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vshr.u64	d16, d16, #64   @ encoding: [0xb0,0x00,0xc0,0xf3]
	%tmp2 = lshr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

; CHECK: vshru_16xi8
define <16 x i8> @vshru_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vshr.u8	q8, q8, #8              @ encoding: [0x70,0x00,0xc8,0xf3]
	%tmp2 = lshr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

; CHECK: vshru_8xi16
define <8 x i16> @vshru_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vshr.u16	q8, q8, #16     @ encoding: [0x70,0x00,0xd0,0xf3]
	%tmp2 = lshr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

; CHECK: vshru_4xi32
define <4 x i32> @vshru_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vshr.u32	q8, q8, #32     @ encoding: [0x70,0x00,0xe0,0xf3]
	%tmp2 = lshr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

; CHECK: vshru_2xi64
define <2 x i64> @vshru_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vshr.u64	q8, q8, #64     @ encoding: [0xf0,0x00,0xc0,0xf3]
	%tmp2 = lshr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}

; CHECK: vshrs_8xi8
define <8 x i8> @vshrs_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshr.s8	d16, d16, #8            @ encoding: [0x30,0x00,0xc8,0xf2
	%tmp2 = ashr <8 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <8 x i8> %tmp2
}

; CHECK: vshrs_4xi16
define <4 x i16> @vshrs_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshr.s16	d16, d16, #16   @ encoding: [0x30,0x00,0xd0,0xf2]
	%tmp2 = ashr <4 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16 >
	ret <4 x i16> %tmp2
}

; CHECK: vshrs_2xi32
define <2 x i32> @vshrs_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshr.s32	d16, d16, #32   @ encoding: [0x30,0x00,0xe0,0xf2]
	%tmp2 = ashr <2 x i32> %tmp1, < i32 32, i32 32 >
	ret <2 x i32> %tmp2
}

; CHECK: vshrs_1xi64
define <1 x i64> @vshrs_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vshr.s64	d16, d16, #64   @ encoding: [0xb0,0x00,0xc0,0xf2]
	%tmp2 = ashr <1 x i64> %tmp1, < i64 64 >
	ret <1 x i64> %tmp2
}

; CHECK: vshrs_16xi8
define <16 x i8> @vshrs_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vshr.s8	q8, q8, #8              @ encoding: [0x70,0x00,0xc8,0xf2]
	%tmp2 = ashr <16 x i8> %tmp1, < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
	ret <16 x i8> %tmp2
}

; CHECK: vshrs_8xi16
define <8 x i16> @vshrs_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vshr.s16	q8, q8, #16     @ encoding: [0x70,0x00,0xd0,0xf2]
	%tmp2 = ashr <8 x i16> %tmp1, < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
	ret <8 x i16> %tmp2
}

; CHECK: vshrs_4xi32
define <4 x i32> @vshrs_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vshr.s32	q8, q8, #32     @ encoding: [0x70,0x00,0xe0,0xf2]
	%tmp2 = ashr <4 x i32> %tmp1, < i32 32, i32 32, i32 32, i32 32 >
	ret <4 x i32> %tmp2
}

; CHECK: vshrs_2xi64
define <2 x i64> @vshrs_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vshr.s64	q8, q8, #64     @ encoding: [0xf0,0x00,0xc0,0xf2]
	%tmp2 = ashr <2 x i64> %tmp1, < i64 64, i64 64 >
	ret <2 x i64> %tmp2
}

declare <8 x i16> @llvm.arm.neon.vshiftls.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vshiftls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vshiftls.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vshlls_8xi8
define <8 x i16> @vshlls_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshll.s8	q8, d16, #7     @ encoding: [0x30,0x0a,0xcf,0xf2]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshiftls.v8i16(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i16> %tmp2
}

; CHECK: vshlls_4xi16
define <4 x i32> @vshlls_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshll.s16	q8, d16, #15    @ encoding: [0x30,0x0a,0xdf,0xf2]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshiftls.v4i32(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i32> %tmp2
}

; CHECK: vshlls_2xi32
define <2 x i64> @vshlls_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshll.s32	q8, d16, #31    @ encoding: [0x30,0x0a,0xff,0xf2]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshiftls.v2i64(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i64> %tmp2
}

declare <8 x i16> @llvm.arm.neon.vshiftlu.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vshiftlu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vshiftlu.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vshllu_8xi8
define <8 x i16> @vshllu_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshll.u8	q8, d16, #7     @ encoding: [0x30,0x0a,0xcf,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshiftlu.v8i16(<8 x i8> %tmp1, <8 x i8> < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >)
	ret <8 x i16> %tmp2
}

; CHECK: vshllu_4xi16
define <4 x i32> @vshllu_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshll.u16	q8, d16, #15    @ encoding: [0x30,0x0a,0xdf,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshiftlu.v4i32(<4 x i16> %tmp1, <4 x i16> < i16 15, i16 15, i16 15, i16 15 >)
	ret <4 x i32> %tmp2
}

; CHECK: vshllu_2xi32
define <2 x i64> @vshllu_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshll.u32	q8, d16, #31    @ encoding: [0x30,0x0a,0xff,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshiftlu.v2i64(<2 x i32> %tmp1, <2 x i32> < i32 31, i32 31 >)
	ret <2 x i64> %tmp2
}

; The following tests use the maximum shift count, so the signedness is
; irrelevant.  Test both signed and unsigned versions.

; CHECK: vshlli_8xi8
define <8 x i16> @vshlli_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshll.i8	q8, d16, #8     @ encoding: [0x20,0x03,0xf2,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vshiftls.v8i16(<8 x i8> %tmp1, <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >)
	ret <8 x i16> %tmp2
}

; CHECK: vshlli_4xi16
define <4 x i32> @vshlli_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshll.i16	q8, d16, #16    @ encoding: [0x20,0x03,0xf6,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vshiftlu.v4i32(<4 x i16> %tmp1, <4 x i16> < i16 16, i16 16, i16 16, i16 16 >)
	ret <4 x i32> %tmp2
}

; CHECK: vshlli_2xi32
define <2 x i64> @vshlli_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshll.i32	q8, d16, #32    @ encoding: [0x20,0x03,0xfa,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vshiftls.v2i64(<2 x i32> %tmp1, <2 x i32> < i32 32, i32 32 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vshiftn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vshiftn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vshrns_8xi16
define <8 x i8> @vshrns_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vshrn.i16	d16, q8, #8     @ encoding: [0x30,0x08,0xc8,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vshiftn.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

; CHECK: vshrns_4xi32
define <4 x i16> @vshrns_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vshrn.i32	d16, q8, #16    @ encoding: [0x30,0x08,0xd0,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

; CHECK: vshrns_2xi64
define <2 x i32> @vshrns_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vshrn.i64	d16, q8, #32    @ encoding: [0x30,0x08,0xe0,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vshiftn.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vrshifts.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vrshls_8xi8
define <8 x i8> @vrshls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrshl.s8	d16, d16, d17   @ encoding: [0xa0,0x05,0x41,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vrshls_4xi16
define <4 x i16> @vrshls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrshl.s16	d16, d16, d17   @ encoding: [0xa0,0x05,0x51,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vrshls_2xi32
define <2 x i32> @vrshls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrshl.s32	d16, d16, d17   @ encoding: [0xa0,0x05,0x61,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vrshls_1xi64
define <1 x i64> @vrshls_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vrshl.s64	d16, d16, d17   @ encoding: [0xa0,0x05,0x71,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vrshiftu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vrshlu_8xi8
define <8 x i8> @vrshlu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrshl.u8	d16, d16, d17   @ encoding: [0xa0,0x05,0x41,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vrshlu_4xi16
define <4 x i16> @vrshlu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrshl.u16	d16, d16, d17   @ encoding: [0xa0,0x05,0x51,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vrshlu_2xi32
define <2 x i32> @vrshlu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrshl.u32	d16, d16, d17   @ encoding: [0xa0,0x05,0x61,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vrshlu_1xi64
define <1 x i64> @vrshlu_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vrshl.u64	d16, d16, d17   @ encoding: [0xa0,0x05,0x71,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vrshls_16xi8
define <16 x i8> @vrshls_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrshl.s8	q8, q8, q9      @ encoding: [0xe0,0x05,0x42,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vrshls_8xi16
define <8 x i16> @vrshls_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrshl.s16	q8, q8, q9      @ encoding: [0xe0,0x05,0x52,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vrshls_4xi32
define <4 x i32> @vrshls_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrshl.s32	q8, q8, q9      @ encoding: [0xe0,0x05,0x62,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vrshls_2xi64
define <2 x i64> @vrshls_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vrshl.s64	q8, q8, q9      @ encoding: [0xe0,0x05,0x72,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vrshlu_16xi8
define <16 x i8> @vrshlu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrshl.u8	q8, q8, q9      @ encoding: [0xe0,0x05,0x42,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vrshlu_8xi16
define <8 x i16> @vrshlu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrshl.u16	q8, q8, q9      @ encoding: [0xe0,0x05,0x52,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vrshlu_4xi32
define <4 x i32> @vrshlu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrshl.u32	q8, q8, q9      @ encoding: [0xe0,0x05,0x62,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vrshlu_2xi64
define <2 x i64> @vrshlu_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vrshl.u64	q8, q8, q9      @ encoding: [0xe0,0x05,0x72,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

; CHECK: vrshrs_8xi8
define <8 x i8> @vrshrs_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vrshr.s8	d16, d16, #8    @ encoding: [0x30,0x02,0xc8,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshifts.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

; CHECK: vrshrs_4xi16
define <4 x i16> @vrshrs_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vrshr.s16	d16, d16, #16   @ encoding: [0x30,0x02,0xd0,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshifts.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

; CHECK: vrshrs_2xi32
define <2 x i32> @vrshrs_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vrshr.s32	d16, d16, #32   @ encoding: [0x30,0x02,0xe0,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshifts.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

; CHECK: vrshrs_1xi64
define <1 x i64> @vrshrs_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vrshr.s64	d16, d16, #64   @ encoding: [0xb0,0x02,0xc0,0xf2]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

; CHECK: vrshru_8xi8
define <8 x i8> @vrshru_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vrshr.u8	d16, d16, #8    @ encoding: [0x30,0x02,0xc8,0xf3]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshiftu.v8i8(<8 x i8> %tmp1, <8 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <8 x i8> %tmp2
}

; CHECK: vrshru_4xi16
define <4 x i16> @vrshru_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vrshr.u16	d16, d16, #16   @ encoding: [0x30,0x02,0xd0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshiftu.v4i16(<4 x i16> %tmp1, <4 x i16> < i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <4 x i16> %tmp2
}

; CHECK: vrshru_2xi32
define <2 x i32> @vrshru_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vrshr.u32	d16, d16, #32   @ encoding: [0x30,0x02,0xe0,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshiftu.v2i32(<2 x i32> %tmp1, <2 x i32> < i32 -32, i32 -32 >)
	ret <2 x i32> %tmp2
}

; CHECK: vrshru_1xi64
define <1 x i64> @vrshru_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vrshr.u64	d16, d16, #64   @ encoding: [0xb0,0x02,0xc0,0xf3]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %tmp1, <1 x i64> < i64 -64 >)
	ret <1 x i64> %tmp2
}

; CHECK: vrshrs_16xi8
define <16 x i8> @vrshrs_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vrshr.s8	q8, q8, #8      @ encoding: [0x70,0x02,0xc8,0xf2]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vrshifts.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

; CHECK: vrshrs_8xi16
define <8 x i16> @vrshrs_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vrshr.s16	q8, q8, #16     @ encoding: [0x70,0x02,0xd0,0xf2]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vrshifts.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

; CHECK: vrshrs_4xi32
define <4 x i32> @vrshrs_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrshr.s32	q8, q8, #32     @ encoding: [0x70,0x02,0xe0,0xf2]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrshifts.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

; CHECK: vrshrs_2xi64
define <2 x i64> @vrshrs_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vrshr.s64	q8, q8, #64     @ encoding: [0xf0,0x02,0xc0,0xf2]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vrshifts.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

; CHECK: vrshru_16xi8
define <16 x i8> @vrshru_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vrshr.u8	q8, q8, #8      @ encoding: [0x70,0x02,0xc8,0xf3]
	%tmp2 = call <16 x i8> @llvm.arm.neon.vrshiftu.v16i8(<16 x i8> %tmp1, <16 x i8> < i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8, i8 -8 >)
	ret <16 x i8> %tmp2
}

; CHECK: vrshru_8xi16
define <8 x i16> @vrshru_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vrshr.u16	q8, q8, #16     @ encoding: [0x70,0x02,0xd0,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %tmp1, <8 x i16> < i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16, i16 -16 >)
	ret <8 x i16> %tmp2
}

; CHECK: vrshru_4xi32
define <4 x i32> @vrshru_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrshr.u32	q8, q8, #32     @ encoding: [0x70,0x02,0xe0,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vrshiftu.v4i32(<4 x i32> %tmp1, <4 x i32> < i32 -32, i32 -32, i32 -32, i32 -32 >)
	ret <4 x i32> %tmp2
}

; CHECK: vrshru_2xi64
define <2 x i64> @vrshru_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vrshr.u64	q8, q8, #64     @ encoding: [0xf0,0x02,0xc0,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vrshiftu.v2i64(<2 x i64> %tmp1, <2 x i64> < i64 -64, i64 -64 >)
	ret <2 x i64> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vrshiftn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vrshrns_8xi16
define <8 x i8> @vrshrns_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vrshrn.i16	d16, q8, #8     @ encoding: [0x70,0x08,0xc8,0xf2]
	%tmp2 = call <8 x i8> @llvm.arm.neon.vrshiftn.v8i8(<8 x i16> %tmp1, <8 x i16> < i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8 >)
	ret <8 x i8> %tmp2
}

; CHECK: vrshrns_4xi32
define <4 x i16> @vrshrns_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrshrn.i32	d16, q8, #16    @ encoding: [0x70,0x08,0xd0,0xf2]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vrshiftn.v4i16(<4 x i32> %tmp1, <4 x i32> < i32 -16, i32 -16, i32 -16, i32 -16 >)
	ret <4 x i16> %tmp2
}

; CHECK: vrshrns_2xi64
define <2 x i32> @vrshrns_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vrshrn.i64	d16, q8, #32    @ encoding: [0x70,0x08,0xe0,0xf2]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vrshiftn.v2i32(<2 x i64> %tmp1, <2 x i64> < i64 -32, i64 -32 >)
	ret <2 x i32> %tmp2
}
