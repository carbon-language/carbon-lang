; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

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
