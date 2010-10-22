; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

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
