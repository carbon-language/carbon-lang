; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; FIXME: The following instructions still require testing:
;  - vand with immediate

; CHECK: vand_8xi8
define <8 x i8> @vand_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vand	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf2]
	%tmp3 = and <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vand_16xi8
define <16 x i8> @vand_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vand	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf2]
	%tmp3 = and <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: veor_8xi8
define <8 x i8> @veor_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: veor	d16, d17, d16           @ encoding: [0xb0,0x01,0x41,0xf3]
	%tmp3 = xor <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: veor_16xi8
define <16 x i8> @veor_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: veor	q8, q8, q9              @ encoding: [0xf2,0x01,0x40,0xf3]
	%tmp3 = xor <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vorr_8xi8
define <8 x i8> @vorr_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vorr	d16, d17, d16           @ encoding: [0xb0,0x01,0x61,0xf2]
	%tmp3 = or <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vorr_16xi8
define <16 x i8> @vorr_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vorr	q8, q8, q9              @ encoding: [0xf2,0x01,0x60,0xf2]
	%tmp3 = or <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}