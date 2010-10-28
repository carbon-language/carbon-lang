; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

declare <8 x i8>  @llvm.arm.neon.vpadd.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpadd.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vpadd_8xi8
define <8 x i8> @vpadd_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpadd.i8	d16, d17, d16   @ encoding: [0xb0,0x0b,0x41,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vpadd_4xi16
define <4 x i16> @vpadd_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpadd.i16	d16, d17, d16   @ encoding: [0xb0,0x0b,0x51,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadd.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpadd_2xi32
define <2 x i32> @vpadd_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpadd.i32	d16, d17, d16   @ encoding: [0xb0,0x0b,0x61,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadd.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vpadd_2xfloat
define <2 x float> @vpadd_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vpadd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x40,0xf3]
	%tmp3 = call <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32>) nounwind readnone

; CHECK: vpaddls_8xi8
define <4 x i16> @vpaddls_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vpaddl.s8	d16, d16        @ encoding: [0x20,0x02,0xf0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

; CHECK: vpaddls_4xi16
define <2 x i32> @vpaddls_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vpaddl.s16	d16, d16        @ encoding: [0x20,0x02,0xf4,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

; CHECK: vpaddls_2xi32
define <1 x i64> @vpaddls_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vpaddl.s32	d16, d16        @ encoding: [0x20,0x02,0xf8,0xf3]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

declare <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32>) nounwind readnone

; CHECK: vpaddlu_8xi8
define <4 x i16> @vpaddlu_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vpaddl.u8	d16, d16        @ encoding: [0xa0,0x02,0xf0,0xf3]
	%tmp2 = call <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8> %tmp1)
	ret <4 x i16> %tmp2
}

; CHECK: vpaddlu_4xi16
define <2 x i32> @vpaddlu_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vpaddl.u16	d16, d16        @ encoding: [0xa0,0x02,0xf4,0xf3]
	%tmp2 = call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %tmp1)
	ret <2 x i32> %tmp2
}

; CHECK: vpaddlu_2xi32
define <1 x i64> @vpaddlu_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vpaddl.u32	d16, d16        @ encoding: [0xa0,0x02,0xf8,0xf3]
	%tmp2 = call <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32> %tmp1)
	ret <1 x i64> %tmp2
}

declare <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32>) nounwind readnone

; CHECK: vpaddls_16xi8
define <8 x i16> @vpaddls_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vpaddl.s8	q8, q8          @ encoding: [0x60,0x02,0xf0,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

; CHECK: vpaddls_8xi16
define <4 x i32> @vpaddls_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vpaddl.s16	q8, q8          @ encoding: [0x60,0x02,0xf4,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

; CHECK: vpaddls_4xi32
define <2 x i64> @vpaddls_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vpaddl.s32	q8, q8          @ encoding: [0x60,0x02,0xf8,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

declare <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) nounwind readnone

; CHECK: vpaddlu_16xi8
define <8 x i16> @vpaddlu_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vpaddl.u8	q8, q8          @ encoding: [0xe0,0x02,0xf0,0xf3]
	%tmp2 = call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %tmp1)
	ret <8 x i16> %tmp2
}

; CHECK: vpaddlu_8xi16
define <4 x i32> @vpaddlu_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vpaddl.u16	q8, q8          @ encoding: [0xe0,0x02,0xf4,0xf3]
	%tmp2 = call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %tmp1)
	ret <4 x i32> %tmp2
}

; CHECK: vpaddlu_4xi32
define <2 x i64> @vpaddlu_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vpaddl.u32	q8, q8          @ encoding: [0xe0,0x02,0xf8,0xf3]
	%tmp2 = call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %tmp1)
	ret <2 x i64> %tmp2
}

declare <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16>, <8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32>, <4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64>, <2 x i32>) nounwind readnone

; CHECK: vpadals_8xi8
define <4 x i16> @vpadals_8xi8(<4 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpadal.s8	d16, d17        @ encoding: [0x21,0x06,0xf0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpadals_4xi16
define <2 x i32> @vpadals_4xi16(<2 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpadal.s16	d16, d17        @ encoding: [0x21,0x06,0xf4,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vpadals_2xi32
define <1 x i64> @vpadals_2xi32(<1 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpadal.s32	d16, d17        @ encoding: [0x21,0x06,0xf8,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

declare <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16>, <8 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32>, <4 x i16>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64>, <2 x i32>) nounwind readnone

; CHECK: vpadalu_8xi8
define <4 x i16> @vpadalu_8xi8(<4 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpadal.u8	d16, d17        @ encoding: [0xa1,0x06,0xf0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16> %tmp1, <8 x i8> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpadalu_4xi16
define <2 x i32> @vpadalu_4xi16(<2 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpadal.u16	d16, d17        @ encoding: [0xa1,0x06,0xf4,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32> %tmp1, <4 x i16> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vpadalu_2xi32
define <1 x i64> @vpadalu_2xi32(<1 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpadal.u32	d16, d17        @ encoding: [0xa1,0x06,0xf8,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64> %tmp1, <2 x i32> %tmp2)
	ret <1 x i64> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16>, <16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32>, <8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64>, <4 x i32>) nounwind readnone

; CHECK: vpadals_16xi8
define <8 x i16> @vpadals_16xi8(<8 x i16>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vpadal.s8	q9, q8          @ encoding: [0x60,0x26,0xf0,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vpadals_8xi16
define <4 x i32> @vpadals_8xi16(<4 x i32>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vpadal.s16	q9, q8          @ encoding: [0x60,0x26,0xf4,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vpadals_4xi32
define <2 x i64> @vpadals_4xi32(<2 x i64>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vpadal.s32	q9, q8          @ encoding: [0x60,0x26,0xf8,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64> %tmp1, <4 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16>, <16 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32>, <8 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64>, <4 x i32>) nounwind readnone

; CHECK: vpadalu_16xi8
define <8 x i16> @vpadalu_16xi8(<8 x i16>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vpadal.u8	q9, q8          @ encoding: [0xe0,0x26,0xf0,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %tmp1, <16 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vpadalu_8xi16
define <4 x i32> @vpadalu_8xi16(<4 x i32>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vpadal.u16	q9, q8          @ encoding: [0xe0,0x26,0xf4,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32> %tmp1, <8 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vpadalu_4xi32
define <2 x i64> @vpadalu_4xi32(<2 x i64>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vpadal.u32	q9, q8          @ encoding: [0xe0,0x26,0xf8,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64> %tmp1, <4 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpmins.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vpmins_8xi8
define <8 x i8> @vpmins_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpmin.s8	d16, d16, d17   @ encoding: [0xb1,0x0a,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmins.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vpmins_4xi16
define <4 x i16> @vpmins_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpmin.s16	d16, d16, d17   @ encoding: [0xb1,0x0a,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpmins_2xi32
define <2 x i32> @vpmins_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpmin.s32	d16, d16, d17   @ encoding: [0xb1,0x0a,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpminu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vpminu_8xi8
define <8 x i8> @vpminu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpmin.u8	d16, d16, d17   @ encoding: [0xb1,0x0a,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpminu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vpminu_4xi16
define <4 x i16> @vpminu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpmin.u16	d16, d16, d17   @ encoding: [0xb1,0x0a,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpminu_2xi32
define <2 x i32> @vpminu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpmin.u32	d16, d16, d17   @ encoding: [0xb1,0x0a,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vpmin_2xfloat
define <2 x float> @vpmin_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vpmin.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x60,0xf3]
	%tmp3 = call <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpmaxs.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vpmaxs_8xi8
define <8 x i8> @vpmaxs_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpmax.s8	d16, d16, d17   @ encoding: [0xa1,0x0a,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmaxs.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vpmaxs_4xi16
define <4 x i16> @vpmaxs_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpmax.s16	d16, d16, d17   @ encoding: [0xa1,0x0a,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpmaxs_2xi32
define <2 x i32> @vpmaxs_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpmax.s32	d16, d16, d17   @ encoding: [0xa1,0x0a,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vpmaxu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vpmaxu_8xi8
define <8 x i8> @vpmaxu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vpmax.u8	d16, d16, d17   @ encoding: [0xa1,0x0a,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vpmaxu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vpmaxu_4xi16
define <4 x i16> @vpmaxu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vpmax.u16	d16, d16, d17   @ encoding: [0xa1,0x0a,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vpmaxu_2xi32
define <2 x i32> @vpmaxu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vpmax.u32	d16, d16, d17   @ encoding: [0xa1,0x0a,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vpmax_2xfloat
define <2 x float> @vpmax_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vpmax.f32	d16, d16, d17   @ encoding: [0xa1,0x0f,0x40,0xf3]
	%tmp3 = call <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}
