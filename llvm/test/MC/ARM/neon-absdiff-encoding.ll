; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

declare <8 x i8>  @llvm.arm.neon.vabds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vabds_8xi8
define <8 x i8> @vabds_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vabd.s8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vabds_4xi16
define <4 x i16> @vabds_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vabd.s16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vabds_2xi32
define <2 x i32> @vabds_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vabd.s32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vabdu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vabdu_8xi8
define <8 x i8> @vabdu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vabd.u8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vabdu_4xi16
define <4 x i16> @vabdu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vabd.u16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vabdu_2xi32
define <2 x i32> @vabdu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vabd.u32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float>, <2 x float>) nounwind readnone

; CHECK: vabd_2xfloat
define <2 x float> @vabd_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vabd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xf3]
	%tmp3 = call <2 x float> @llvm.arm.neon.vabds.v2f32(<2 x float> %tmp1, <2 x float> %tmp2)
	ret <2 x float> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vabds_16xi8
define <16 x i8> @vabds_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vabd.s8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vabds_8xi16
define <8 x i16> @vabds_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vabd.s16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vabds_4xi32
define <4 x i32> @vabds_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vabd.s32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vabdu_16xi8
define <16 x i8> @vabdu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vabd.u8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vabdu_8xi16
define <8 x i16> @vabdu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vabd.u16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vabdu_4xi32
define <4 x i32> @vabdu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vabd.u32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float>, <4 x float>) nounwind readnone

; CHECK: vabd_4xfloat
define <4 x float> @vabd_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vabd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xf3]
	%tmp3 = call <4 x float> @llvm.arm.neon.vabds.v4f32(<4 x float> %tmp1, <4 x float> %tmp2)
	ret <4 x float> %tmp3
}

; CHECK: vabdls_8xi8
define <8 x i16> @vabdls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vabdl.s8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	%tmp4 = zext <8 x i8> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vabdls_4xi16
define <4 x i32> @vabdls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vabdl.s16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	%tmp4 = zext <4 x i16> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vabdls_2xi32
define <2 x i64> @vabdls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vabdl.s32	q8, d16, d17    @ encoding: [0xa1,0x07,0xe0,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	%tmp4 = zext <2 x i32> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

; CHECK: vabdlu_8xi8
define <8 x i16> @vabdlu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vabdl.u8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	%tmp4 = zext <8 x i8> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vabdlu_4xi16
define <4 x i32> @vabdlu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vabdl.u16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	%tmp4 = zext <4 x i16> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vabdlu_2xi3
define <2 x i64> @vabdlu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	%tmp4 = zext <2 x i32> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

; CHECK: vabas_8xi8
define <8 x i8> @vabas_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vaba.s8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xf2]
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

; CHECK: vabas_4xi16
define <4 x i16> @vabas_4xi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vaba.s16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xf2]
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

; CHECK: vabas_2xi32
define <2 x i32> @vabas_2xi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vaba.s32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xf2]
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

; CHECK: vabau_8xi8
define <8 x i8> @vabau_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vaba.u8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xf3]
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

; CHECK: vabau_4xi16
define <4 x i16> @vabau_4xi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vaba.u16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xf3]
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

; CHECK: vabau_2xi32
define <2 x i32> @vabau_2xi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vaba.u32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xf3]
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

; CHECK: vabas_16xi8
define <16 x i8> @vabas_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vaba.s8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xf2]
	%tmp4 = call <16 x i8> @llvm.arm.neon.vabds.v16i8(<16 x i8> %tmp2, <16 x i8> %tmp3)
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

; CHECK: vabas_8xi16
define <8 x i16> @vabas_8xi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
; CHECK: vaba.s16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xf2]
	%tmp4 = call <8 x i16> @llvm.arm.neon.vabds.v8i16(<8 x i16> %tmp2, <8 x i16> %tmp3)
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vabas_4xi32
define <4 x i32> @vabas_4xi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
; CHECK: vaba.s32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xf2]
	%tmp4 = call <4 x i32> @llvm.arm.neon.vabds.v4i32(<4 x i32> %tmp2, <4 x i32> %tmp3)
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vabau_16xi8
define <16 x i8> @vabau_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vaba.u8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xf3]
	%tmp4 = call <16 x i8> @llvm.arm.neon.vabdu.v16i8(<16 x i8> %tmp2, <16 x i8> %tmp3)
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

; CHECK: vabau_8xi16
define <8 x i16> @vabau_8xi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
; CHECK: vaba.u16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xf3]
	%tmp4 = call <8 x i16> @llvm.arm.neon.vabdu.v8i16(<8 x i16> %tmp2, <8 x i16> %tmp3)
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vabau_4xi32
define <4 x i32> @vabau_4xi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
; CHECK: vaba.u32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xf3]
	%tmp4 = call <4 x i32> @llvm.arm.neon.vabdu.v4i32(<4 x i32> %tmp2, <4 x i32> %tmp3)
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vabals_8xi8
define <8 x i16> @vabals_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vabal.s8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xf2]
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabds.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = zext <8 x i8> %tmp4 to <8 x i16>
	%tmp6 = add <8 x i16> %tmp1, %tmp5
	ret <8 x i16> %tmp6
}

; CHECK: vabals_4xi16
define <4 x i32> @vabals_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vabal.s16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xf2]
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabds.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = zext <4 x i16> %tmp4 to <4 x i32>
	%tmp6 = add <4 x i32> %tmp1, %tmp5
	ret <4 x i32> %tmp6
}

; CHECK: vabals_2xi32
define <2 x i64> @vabals_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vabal.s32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xf2]
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabds.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = zext <2 x i32> %tmp4 to <2 x i64>
	%tmp6 = add <2 x i64> %tmp1, %tmp5
	ret <2 x i64> %tmp6
}

; CHECK: vabalu_8xi8
define <8 x i16> @vabalu_8xi8(<8 x i16>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vabal.u8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xf3]
	%tmp4 = call <8 x i8> @llvm.arm.neon.vabdu.v8i8(<8 x i8> %tmp2, <8 x i8> %tmp3)
	%tmp5 = zext <8 x i8> %tmp4 to <8 x i16>
	%tmp6 = add <8 x i16> %tmp1, %tmp5
	ret <8 x i16> %tmp6
}

; CHECK: vabalu_4xi16
define <4 x i32> @vabalu_4xi16(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: 	vabal.u16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xf3]
	%tmp4 = call <4 x i16> @llvm.arm.neon.vabdu.v4i16(<4 x i16> %tmp2, <4 x i16> %tmp3)
	%tmp5 = zext <4 x i16> %tmp4 to <4 x i32>
	%tmp6 = add <4 x i32> %tmp1, %tmp5
	ret <4 x i32> %tmp6
}

; CHECK: vabalu_2xi32
define <2 x i64> @vabalu_2xi32(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vabal.u32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xf3]
	%tmp4 = call <2 x i32> @llvm.arm.neon.vabdu.v2i32(<2 x i32> %tmp2, <2 x i32> %tmp3)
	%tmp5 = zext <2 x i32> %tmp4 to <2 x i64>
	%tmp6 = add <2 x i64> %tmp1, %tmp5
	ret <2 x i64> %tmp6
}
