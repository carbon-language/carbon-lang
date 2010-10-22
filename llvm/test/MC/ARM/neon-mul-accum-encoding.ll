; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; CHECK: vmla_8xi8
define <8 x i8> @vmla_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8> * %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vmla.i8	d16, d18, d17           @ encoding: [0xa1,0x09,0x42,0xf2]
	%tmp4 = mul <8 x i8> %tmp2, %tmp3
	%tmp5 = add <8 x i8> %tmp1, %tmp4
	ret <8 x i8> %tmp5
}

; CHECK: vmla_4xi16
define <4 x i16> @vmla_4xi16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
; CHECK: vmla.i16	d16, d18, d17   @ encoding: [0xa1,0x09,0x52,0xf2]
	%tmp4 = mul <4 x i16> %tmp2, %tmp3
	%tmp5 = add <4 x i16> %tmp1, %tmp4
	ret <4 x i16> %tmp5
}

; CHECK: vmla_2xi32
define <2 x i32> @vmla_2xi32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
; CHECK: vmla.i32	d16, d18, d17   @ encoding: [0xa1,0x09,0x62,0xf2]
	%tmp4 = mul <2 x i32> %tmp2, %tmp3
	%tmp5 = add <2 x i32> %tmp1, %tmp4
	ret <2 x i32> %tmp5
}

; CHECK: vmla_2xfloat
define <2 x float> @vmla_2xfloat(<2 x float>* %A, <2 x float>* %B, <2 x float>* %C) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = load <2 x float>* %C
; CHECK: vmla.f32	d16, d18, d17   @ encoding: [0xb1,0x0d,0x42,0xf2]
	%tmp4 = fmul <2 x float> %tmp2, %tmp3
	%tmp5 = fadd <2 x float> %tmp1, %tmp4
	ret <2 x float> %tmp5
}

; CHECK: vmla_16xi8
define <16 x i8> @vmla_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8> * %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vmla.i8	q9, q8, q10             @ encoding: [0xe4,0x29,0x40,0xf2]
	%tmp4 = mul <16 x i8> %tmp2, %tmp3
	%tmp5 = add <16 x i8> %tmp1, %tmp4
	ret <16 x i8> %tmp5
}

; CHECK: vmla_8xi16
define <8 x i16> @vmla_8xi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
; CHECK: vmla.i16	q9, q8, q10     @ encoding: [0xe4,0x29,0x50,0xf2]
	%tmp4 = mul <8 x i16> %tmp2, %tmp3
	%tmp5 = add <8 x i16> %tmp1, %tmp4
	ret <8 x i16> %tmp5
}

; CHECK: vmla_4xi32
define <4 x i32> @vmla_4xi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
; CHECK: vmla.i32	q9, q8, q10     @ encoding: [0xe4,0x29,0x60,0xf2]
	%tmp4 = mul <4 x i32> %tmp2, %tmp3
	%tmp5 = add <4 x i32> %tmp1, %tmp4
	ret <4 x i32> %tmp5
}

; CHECK: vmla_4xfloat
define <4 x float> @vmla_4xfloat(<4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = load <4 x float>* %C
; CHECK: vmla.f32	q9, q8, q10     @ encoding: [0xf4,0x2d,0x40,0xf2]
	%tmp4 = fmul <4 x float> %tmp2, %tmp3
	%tmp5 = fadd <4 x float> %tmp1, %tmp4
	ret <4 x float> %tmp5
}
