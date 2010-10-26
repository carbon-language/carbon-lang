; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

; CHECK: vshls_8xi8
define <8 x i8> @vshls_8xi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
; CHECK: vshl.u8	d16, d17, d16           @ encoding: [0xa1,0x04,0x40,0xf3]
	%tmp3 = shl <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

; CHECK: vshls_4xi16
define <4 x i16> @vshls_4xi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
; CHECK: vshl.u16	d16, d17, d16   @ encoding: [0xa1,0x04,0x50,0xf3]
	%tmp3 = shl <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

; CHECK: vshls_2xi32
define <2 x i32> @vshls_2xi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
; CHECK: vshl.u32	d16, d17, d16   @ encoding: [0xa1,0x04,0x60,0xf3]
	%tmp3 = shl <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

; CHECK: vshls_1xi64
define <1 x i64> @vshls_1xi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
; CHECK: vshl.u64	d16, d17, d16   @ encoding: [0xa1,0x04,0x70,0xf3]
	%tmp3 = shl <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

; CHECK: vshli_8xi8
define <8 x i8> @vshli_8xi8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vshl.i8	d16, d16, #7            @ encoding: [0x30,0x05,0xcf,0xf2]
	%tmp2 = shl <8 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <8 x i8> %tmp2
}

; CHECK: vshli_4xi16
define <4 x i16> @vshli_4xi16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vshl.i16	d16, d16, #15   @ encoding: [0x30,0x05,0xdf,0xf2
	%tmp2 = shl <4 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15 >
	ret <4 x i16> %tmp2
}

; CHECK: vshli_2xi32
define <2 x i32> @vshli_2xi32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vshl.i32	d16, d16, #31   @ encoding: [0x30,0x05,0xff,0xf2]
	%tmp2 = shl <2 x i32> %tmp1, < i32 31, i32 31 >
	ret <2 x i32> %tmp2
}

; CHECK: vshli_1xi64
define <1 x i64> @vshli_1xi64(<1 x i64>* %A) nounwind {
	%tmp1 = load <1 x i64>* %A
; CHECK: vshl.i64	d16, d16, #63   @ encoding: [0xb0,0x05,0xff,0xf2]
	%tmp2 = shl <1 x i64> %tmp1, < i64 63 >
	ret <1 x i64> %tmp2
}

; CHECK: vshls_16xi8
define <16 x i8> @vshls_16xi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
; CHECK: vshl.u8	q8, q8, q9              @ encoding: [0xe0,0x04,0x42,0xf3]
	%tmp3 = shl <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

; CHECK: vshls_8xi16
define <8 x i16> @vshls_8xi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shl <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

; CHECK: vshls_4xi32
define <4 x i32> @vshls_4xi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
; CHECK: vshl.u32	q8, q8, q9      @ encoding: [0xe0,0x04,0x62,0xf3]
	%tmp3 = shl <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

; CHECK: vshls_2xi64
define <2 x i64> @vshls_2xi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
; CHECK: vshl.u64	q8, q8, q9      @ encoding: [0xe0,0x04,0x72,0xf3]
	%tmp3 = shl <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

; CHECK: vshli_16xi8
define <16 x i8> @vshli_16xi8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vshl.i8	q8, q8, #7              @ encoding: [0x70,0x05,0xcf,0xf2]
	%tmp2 = shl <16 x i8> %tmp1, < i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7 >
	ret <16 x i8> %tmp2
}

; CHECK: vshli_8xi16
define <8 x i16> @vshli_8xi16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vshl.i16	q8, q8, #15     @ encoding: [0x70,0x05,0xdf,0xf2]
	%tmp2 = shl <8 x i16> %tmp1, < i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15 >
	ret <8 x i16> %tmp2
}

; CHECK: vshli_4xi32
define <4 x i32> @vshli_4xi32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vshl.i32	q8, q8, #31     @ encoding: [0x70,0x05,0xff,0xf2]
	%tmp2 = shl <4 x i32> %tmp1, < i32 31, i32 31, i32 31, i32 31 >
	ret <4 x i32> %tmp2
}

; CHECK: vshli_2xi64
define <2 x i64> @vshli_2xi64(<2 x i64>* %A) nounwind {
	%tmp1 = load <2 x i64>* %A
; CHECK: vshl.i64	q8, q8, #63     @ encoding: [0xf0,0x05,0xff,0xf2]
	%tmp2 = shl <2 x i64> %tmp1, < i64 63, i64 63 >
	ret <2 x i64> %tmp2
}
