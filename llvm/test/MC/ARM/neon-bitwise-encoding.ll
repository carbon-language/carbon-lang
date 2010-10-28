; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; XFAIL: *

; FIXME: The following instructions still require testing:
;  - vand with immediate, vorr with immediate
;  - both vbit and vbif

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

; CHECK: vbic_8xi8
define <8 x i8> @vbic_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vbic	d16, d17, d16           @ encoding: [0xb0,0x01,0x51,0xf2]
	%tmp3 = xor <8 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = and <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

; CHECK: vbic_16xi8
define <16 x i8> @vbic_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vbic	q8, q8, q9              @ encoding: [0xf2,0x01,0x50,0xf2]
	%tmp3 = xor <16 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = and <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

; CHECK: vorn_8xi8
define <8 x i8> @vorn_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vorn	d16, d17, d16           @ encoding: [0xb0,0x01,0x71,0xf2]
	%tmp3 = xor <8 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <8 x i8> %tmp1, %tmp3
	ret <8 x i8> %tmp4
}

; CHECK: vorn_16xi8
define <16 x i8> @vorn_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vorn	q8, q8, q9              @ encoding: [0xf2,0x01,0x70,0xf2]
	%tmp3 = xor <16 x i8> %tmp2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp4 = or <16 x i8> %tmp1, %tmp3
	ret <16 x i8> %tmp4
}

; CHECK: vmvn_8xi8
define <8 x i8> @vmvn_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vmvn	d16, d16                @ encoding: [0xa0,0x05,0xf0,0xf3]
	%tmp2 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <8 x i8> %tmp2
}

; CHECK: vmvn_16xi8
define <16 x i8> @vmvn_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vmvn	q8, q8                  @ encoding: [0xe0,0x05,0xf0,0xf3]
	%tmp2 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	ret <16 x i8> %tmp2
}

; CHECK: vbsl_8xi8
define <8 x i8> @vbsl_8xi8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
; CHECK: vbsl	d18, d17, d16           @ encoding: [0xb0,0x21,0x51,0xf3]
	%tmp4 = and <8 x i8> %tmp1, %tmp2
	%tmp5 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <8 x i8> %tmp5, %tmp3
	%tmp7 = or <8 x i8> %tmp4, %tmp6
	ret <8 x i8> %tmp7
}

; CHECK: vbsl_16xi8
define <16 x i8> @vbsl_16xi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
; CHECK: vbsl	q8, q10, q9             @ encoding: [0xf2,0x01,0x54,0xf3]
	%tmp4 = and <16 x i8> %tmp1, %tmp2
	%tmp5 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <16 x i8> %tmp5, %tmp3
	%tmp7 = or <16 x i8> %tmp4, %tmp6
	ret <16 x i8> %tmp7
}
