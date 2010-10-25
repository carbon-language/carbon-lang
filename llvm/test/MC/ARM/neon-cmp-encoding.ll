; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; CHECK: vceq_8xi8
define <8 x i8> @vceq_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vceq.i8	d16, d16, d17           @ encoding: [0xb1,0x08,0x40,0xf3]
	%tmp3 = icmp eq <8 x i8> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

; CHECK: vceq_4xi16
define <4 x i16> @vceq_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vceq.i16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf3]
	%tmp3 = icmp eq <4 x i16> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

; CHECK: vceq_2xi32
define <2 x i32> @vceq_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vceq.i32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf3]
	%tmp3 = icmp eq <2 x i32> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vceq_2xfloat
define <2 x i32> @vceq_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vceq.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf2]
	%tmp3 = fcmp oeq <2 x float> %tmp1, %tmp2
  %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

; CHECK: vceq_16xi8
define <16 x i8> @vceq_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vceq.i8	q8, q8, q9              @ encoding: [0xf2,0x08,0x40,0xf3]
	%tmp3 = icmp eq <16 x i8> %tmp1, %tmp2
  %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

; CHECK: vceq_8xi16
define <8 x i16> @vceq_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vceq.i16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf3]
	%tmp3 = icmp eq <8 x i16> %tmp1, %tmp2
  %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

; CHECK: vceq_4xi32
define <4 x i32> @vceq_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vceq.i32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf3]
	%tmp3 = icmp eq <4 x i32> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; CHECK: vceq_4xfloat
define <4 x i32> @vceq_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vceq.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf2]
	%tmp3 = fcmp oeq <4 x float> %tmp1, %tmp2
  %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

; FIXME: We cannot currently test the vceq with immediate zero, because we do
; not code generate it at this time.

