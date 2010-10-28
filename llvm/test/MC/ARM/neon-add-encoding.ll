; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; CHECK: vadd_8xi8
define <8 x i8> @vadd_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {                                                                          
  %tmp1 = load <8 x i8>* %A
  %tmp2 = load <8 x i8>* %B
; CHECK: vadd.i8	d16, d17, d16           @ encoding: [0xa0,0x08,0x41,0xf2]
  %tmp3 = add <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vadd_4xi16
define <4 x i16> @vadd_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {                                                                          
  %tmp1 = load <4 x i16>* %A
  %tmp2 = load <4 x i16>* %B
; CHECK: vadd.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf2]
  %tmp3 = add <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vadd_1xi64
define <1 x i64> @vadd_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {                                                                          
  %tmp1 = load <1 x i64>* %A
  %tmp2 = load <1 x i64>* %B
; CHECK: vadd.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf2]
  %tmp3 = add <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

; CHECK: vadd_2xi32
define <2 x i32> @vadd_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {                                                                          
  %tmp1 = load <2 x i32>* %A
  %tmp2 = load <2 x i32>* %B
; CHECK: vadd.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf2]
  %tmp3 = add <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vadd_2xfloat
define <2 x float> @vadd_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {                                                                          
  %tmp1 = load <2 x float>* %A
  %tmp2 = load <2 x float>* %B
; CHECK: vadd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x40,0xf2]
  %tmp3 = fadd <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

; CHECK: vadd_4xfloat
define <4 x float> @vadd_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vadd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x40,0xf2]
	%tmp3 = fadd <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

; CHECK: vaddls_8xi8
define <8 x i16> @vaddls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vaddl.s8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf2]
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vaddls_4xi16
define <4 x i32> @vaddls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vaddl.s16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf2]
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vaddls_2xi32
define <2 x i64> @vaddls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vaddl.s32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf2]
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

; CHECK: vaddlu_8xi8
define <8 x i16> @vaddlu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vaddl.u8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf3]
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vaddlu_4xi16
define <4 x i32> @vaddlu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vaddl.u16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf3]
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vaddlu_2xi32
define <2 x i64> @vaddlu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vaddl.u32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf3]
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

; CHECK: vaddws_8xi8
define <8 x i16> @vaddws_8xi8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vaddw.s8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf2]
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

; CHECK: vaddws_4xi16
define <4 x i32> @vaddws_4xi16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vaddw.s16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf2]
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

; CHECK: vaddws_2xi32
define <2 x i64> @vaddws_2xi32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vaddw.s32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf2]
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

; CHECK: vaddwu_8xi8
define <8 x i16> @vaddwu_8xi8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
; CHECK: vaddw.u8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf3]
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

; CHECK: vaddwu_4xi16
define <4 x i32> @vaddwu_4xi16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
; CHECK: vaddw.u16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf3]
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

; CHECK: vaddwu_2xi32
define <2 x i64> @vaddwu_2xi32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
; CHECK: vaddw.u32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf3]
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

declare <8 x i8>  @llvm.arm.neon.vhadds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhadds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vhadds_8xi8
define <8 x i8> @vhadds_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhadds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vhadds_4xi16
define <4 x i16> @vhadds_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vhadds_2xi32
define <2 x i32> @vhadds_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhadds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vhaddu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vhaddu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vhaddu_8xi8
define <8 x i8> @vhaddu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vhaddu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vhaddu_4xi16
define <4 x i16> @vhaddu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vhaddu_2xi32
define <2 x i32> @vhaddu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vhaddu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vhadds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vhadds_16xi8
define <16 x i8> @vhadds_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vhadds_8xi16
define <8 x i16> @vhadds_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vhadds_4xi32
define <4 x i32> @vhadds_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vhadds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vhaddu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vhaddu_16xi8
define <16 x i8> @vhaddu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vhaddu_8xi16
define <8 x i16> @vhaddu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vhaddu_4xi32
define <4 x i32> @vhaddu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vhaddu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vrhadds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrhadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrhadds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vrhadds_8xi8
define <8 x i8> @vrhadds_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrhadds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vrhadds_4xi16
define <4 x i16> @vrhadds_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrhadds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vrhadds_2xi32
define <2 x i32> @vrhadds_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrhadds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vrhaddu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vrhaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vrhaddu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; CHECK: vrhaddu_8xi8
define <8 x i8> @vrhaddu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vrhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vrhaddu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vrhaddu_4xi16
define <4 x i16> @vrhaddu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vrhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vrhaddu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vrhaddu_2xi32
define <2 x i32> @vrhaddu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vrhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vrhaddu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vrhadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrhadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrhadds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vrhadds_16xi8
define <16 x i8> @vrhadds_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrhadds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vrhadds_8xi16
define <8 x i16> @vrhadds_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrhadds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vrhadds_4xi32
define <4 x i32> @vrhadds_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrhadds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vrhaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vrhaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vrhaddu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

; CHECK: vrhaddu_16xi8
define <16 x i8> @vrhaddu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vrhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vrhaddu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vrhaddu_8xi16
define <8 x i16> @vrhaddu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vrhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vrhaddu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vrhaddu_4xi32
define <4 x i32> @vrhaddu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vrhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vrhaddu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqadds.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqadds.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqadds.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vqadds_8xi8
define <8 x i8> @vqadds_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqadd.s8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqadds.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vqadds_4xi16
define <4 x i16> @vqadds_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqadd.s16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqadds.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqadds_2xi32
define <2 x i32> @vqadds_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqadd.s32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqadds.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vqadds_1xi64
define <1 x i64> @vqadds_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqadd.s64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf2]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vqaddu.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqaddu.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqaddu.v2i32(<2 x i32>, <2 x i32>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64>, <1 x i64>) nounwind readnone

; CHECK: vqaddu_8xi8
define <8 x i8> @vqaddu_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vqadd.u8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vqaddu.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vqaddu_4xi16
define <4 x i16> @vqaddu_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vqadd.u16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqaddu.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vqaddu_2xi32
define <2 x i32> @vqaddu_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vqadd.u32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqaddu.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

; CHECK: vqaddu_1xi64
define <1 x i64> @vqaddu_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vqadd.u64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf3]
	%tmp3 = call <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64> %tmp1, <1 x i64> %tmp2)
	ret <1 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vqadds.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqadds.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vqadds_16xi8
define <16 x i8> @vqadds_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqadd.s8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf2]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqadds.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vqadds_8xi16
define <8 x i16> @vqadds_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqadd.s16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf2]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqadds.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqadds_4xi32
define <4 x i32> @vqadds_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqadd.s32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf2]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqadds.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vqadds_2xi64
define <2 x i64> @vqadds_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqadd.s64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf2]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqadds.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <16 x i8> @llvm.arm.neon.vqaddu.v16i8(<16 x i8>, <16 x i8>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqaddu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqaddu.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vqaddu.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vqaddu_16xi8
define <16 x i8> @vqaddu_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vqadd.u8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vqaddu.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

; CHECK: vqaddu_8xi16
define <8 x i16> @vqaddu_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vqadd.u16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf3]
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqaddu.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

; CHECK: vqaddu_4xi32
define <4 x i32> @vqaddu_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vqadd.u32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf3]
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqaddu.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

; CHECK: vqaddu_2xi64
define <2 x i64> @vqaddu_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vqadd.u64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf3]
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqaddu.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i64> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vaddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vaddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vaddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vaddhn_8xi16
define <8 x i8> @vaddhn_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vaddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf2]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vaddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vaddhn_4xi32
define <4 x i16> @vaddhn_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vaddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf2]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vaddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vaddhn_2xi64
define <2 x i32> @vaddhn_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vaddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf2]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vaddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vraddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

; CHECK: vraddhn_8xi16
define <8 x i8> @vraddhn_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vraddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vraddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vraddhn_4xi32
define <4 x i16> @vraddhn_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vraddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf3]
	%tmp3 = call <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

; CHECK: vraddhn_2xi64
define <2 x i32> @vraddhn_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vraddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf3]
	%tmp3 = call <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}
