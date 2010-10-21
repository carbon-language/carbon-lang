; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; CHECK: vmul_8xi8
define <8 x i8> @vmul_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmul.i8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf2]
	%tmp3 = mul <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vmul_4xi16
define <4 x i16> @vmul_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
	%tmp3 = mul <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vmul_2xi32
define <2 x i32> @vmul_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
	%tmp3 = mul <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vmul_2xfloat
define <2 x float> @vmul_2xfloat(<2 x float>* %A, <2 x float>* %B) nounwind {
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
; CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
	%tmp3 = fmul <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

; CHECK: vmul_16xi8
define <16 x i8> @vmul_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmul.i8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf2]
	%tmp3 = mul <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vmul_8xi16
define <8 x i16> @vmul_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
; CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
	%tmp3 = mul <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

; CHECK: vmul_4xi32
define <4 x i32> @vmul_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
	%tmp3 = mul <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

; CHECK: vmul_4xfloat
define <4 x float> @vmul_4xfloat(<4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
; CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
	%tmp3 = fmul <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8>  @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

; CHECK: vmulp_8xi8
define <8 x i8> @vmulp_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vmul.p8	d16, d16, d17           @ encoding: [0xb1,0x09,0x40,0xf3]
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

; CHECK: vmulp_16xi8
define <16 x i8> @vmulp_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vmul.p8	q8, q8, q9              @ encoding: [0xf2,0x09,0x40,0xf3]
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}
