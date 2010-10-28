; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; CHECK: vsub_8xi8
define <8 x i8> @vsub_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vsub.i8	d16, d17, d16           @ encoding: [0xa0,0x08,0x41,0xf3]
	%tmp3 = sub <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vsub_4xi16
define <4 x i16> @vsub_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vsub.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf3]
	%tmp3 = sub <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vsub_2xi32
define <2 x i32> @vsub_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vsub.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf3]
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sub <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vsub_1xi64
define <1 x i64> @vsub_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vsub.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf3]
	%tmp3 = sub <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

; CHECK: vsub_2xifloat
define <2 x float> @vsub_2xifloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vsub.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xf2]
	%tmp3 = fsub <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

; CHECK: vsub_16xi8
define <16 x i8> @vsub_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vsub.i8	q8, q8, q9              @ encoding: [0xe2,0x08,0x40,0xf3]
	%tmp3 = sub <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vsub_8xi16
define <8 x i16> @vsub_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vsub.i16	q8, q8, q9      @ encoding: [0xe2,0x08,0x50,0xf3]
	%tmp3 = sub <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

; CHECK: vsub_4xi32
define <4 x i32> @vsub_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vsub.i32	q8, q8, q9      @ encoding: [0xe2,0x08,0x60,0xf3]
	%tmp3 = sub <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

; CHECK: vsub_2xi64
define <2 x i64> @vsub_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vsub.i64	q8, q8, q9      @ encoding: [0xe2,0x08,0x70,0xf3]
	%tmp3 = sub <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

; CHECK: vsub_4xfloat
define <4 x float> @vsub_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vsub.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xf2]
	%tmp3 = fsub <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

; CHECK: vsubls_8xi8
define <8 x i16> @vsubls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vsubl.s8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf2]
	%tmp5 = sub <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vsubls_4xi16
define <4 x i32> @vsubls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vsubl.s16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf2]
	%tmp5 = sub <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vsubls_2xi32
define <2 x i64> @vsubls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vsubl.s32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf2]
	%tmp5 = sub <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

; CHECK: vsublu_8xi8
define <8 x i16> @vsublu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vsubl.u8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf3]
	%tmp5 = sub <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vsublu_4xi16
define <4 x i32> @vsublu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vsubl.u16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf3]
	%tmp5 = sub <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vsublu_2xi32
define <2 x i64> @vsublu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vsubl.u32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf3]
	%tmp5 = sub <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

; CHECK: vsubws_8xi8
define <8 x i16> @vsubws_8xi8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vsubw.s8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf2]
	%tmp4 = sub <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

; CHECK: vsubws_4xi16
define <4 x i32> @vsubws_4xi16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vsubw.s16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf2]
	%tmp4 = sub <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

; CHECK: vsubws_2xi32
define <2 x i64> @vsubws_2xi32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vsubw.s32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf2]
	%tmp4 = sub <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

; CHECK: vsubwu_8xi8
define <8 x i16> @vsubwu_8xi8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vsubw.u8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf3]
	%tmp4 = sub <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

; CHECK: vsubwu_4xi16
define <4 x i32> @vsubwu_4xi16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vsubw.u16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf3]
	%tmp4 = sub <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

; CHECK: vsubwu_2xi32
define <2 x i64> @vsubwu_2xi32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vsubw.u32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf3]
	%tmp4 = sub <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

declare <8 x i8>  @llvm.arm.neon.vhsubs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vhsubs_8xi8
define <8 x i8> @vhsubs_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vhsub.s8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhsubs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vhsubs_4xi16
define <4 x i16> @vhsubs_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vhsub.s16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vhsubs_2xi32
define <2 x i32> @vhsubs_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vhsub.s32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vhsubu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vhsubu_8xi8
define <8 x i8> @vhsubu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vhsub.u8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhsubu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vhsubu_4xi16
define <4 x i16> @vhsubu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vhsub.u16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vhsubu_2xi32
define <2 x i32> @vhsubu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vhsub.u32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vhsubs_16xi8
define <16 x i8> @vhsubs_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vhsub.s8	q8, q8, q9      @ encoding: [0xe2,0x02,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vhsubs_8xi16
define <8 x i16> @vhsubs_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vhsub.s16	q8, q8, q9      @ encoding: [0xe2,0x02,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vhsubs_4xi32
define <4 x i32> @vhsubs_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vhsub.s32	q8, q8, q9      @ encoding: [0xe2,0x02,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqsubs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqsubs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqsubs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vqsubs_8xi8
define <8 x i8> @vqsubs_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqsub.s8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqsubs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vqsubs_4xi16
define <4 x i16> @vqsubs_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqsub.s16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqsubs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqsubs_2xi32
define <2 x i32> @vqsubs_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqsub.s32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqsubs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vqsubs_1xi64
define <1 x i64> @vqsubs_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqsub.s64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqsubu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqsubu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqsubu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vqsubu_8xi8
define <8 x i8> @vqsubu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqsub.u8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqsubu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vqsubu_4xi16
define <4 x i16> @vqsubu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqsub.u16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqsubu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqsubu_2xi32
define <2 x i32> @vqsubu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqsub.u32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqsubu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vqsubu_1xi64
define <1 x i64> @vqsubu_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqsub.u64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vqsubs.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqsubs.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vqsubs_16xi8
define <16 x i8> @vqsubs_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqsub.s8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqsubs.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vqsubs_8xi16
define <8 x i16> @vqsubs_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqsub.s16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqsubs.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqsubs_4xi32
define <4 x i32> @vqsubs_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqsub.s32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqsubs.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vqsubs_2xi64
define <2 x i64> @vqsubs_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqsub.s64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqsubs.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vqsubu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqsubu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqsubu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqsubu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vqsubu_16xi8
define <16 x i8> @vqsubu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqsub.u8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqsubu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vqsubu_8xi16
define <8 x i16> @vqsubu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqsub.u16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqsubu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqsubu_4xi32
define <4 x i32> @vqsubu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqsub.u32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqsubu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vqsubu_2xi64
define <2 x i64> @vqsubu_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqsub.u64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqsubu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vsubhn_8xi16
define <8 x i8> @vsubhn_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vsubhn_4xi32
define <4 x i16> @vsubhn_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vsubhn_2xi64
define <2 x i32> @vsubhn_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vrsubhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrsubhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrsubhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vrsubhn_8xi16
define <8 x i8> @vrsubhn_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrsubhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vrsubhn_4xi32
define <4 x i16> @vrsubhn_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrsubhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vrsubhn_2xi64
define <2 x i32> @vrsubhn_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vrsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrsubhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}
