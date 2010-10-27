; RUN: llc -show-mc-encoding -march=arm -mcpu=cortex-a8 -mattr=+neon < %s | FileCheck %s

define <8 x i8> @test_vrev64D8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vrev64.8	d16, d16        @ encoding: [0x20,0x00,0xf0,0xf3]
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev64D16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vrev64.16	d16, d16        @ encoding: [0x20,0x00,0xf4,0xf3]
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
	ret <4 x i16> %tmp2
}

define <2 x i32> @test_vrev64D32(<2 x i32>* %A) nounwind {
	%tmp1 = load <2 x i32>* %A
; CHECK: vrev64.32	d16, d16        @ encoding: [0x20,0x00,0xf8,0xf3]
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x i32> %tmp2
}

define <16 x i8> @test_vrev64Q8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vrev64.8	q8, q8          @ encoding: [0x60,0x00,0xf0,0xf3]
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev64Q16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vrev64.16	q8, q8          @ encoding: [0x60,0x00,0xf4,0xf3]
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i16> %tmp2
}

define <4 x i32> @test_vrev64Q32(<4 x i32>* %A) nounwind {
	%tmp1 = load <4 x i32>* %A
; CHECK: vrev64.32	q8, q8          @ encoding: [0x60,0x00,0xf8,0xf3]
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i32> %tmp2
}

define <8 x i8> @test_vrev32D8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vrev32.8	d16, d16        @ encoding: [0xa0,0x00,0xf0,0xf3]
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev32D16(<4 x i16>* %A) nounwind {
	%tmp1 = load <4 x i16>* %A
; CHECK: vrev32.16	d16, d16        @ encoding: [0xa0,0x00,0xf4,0xf3]
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i16> %tmp2
}

define <16 x i8> @test_vrev32Q8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vrev32.8	q8, q8          @ encoding: [0xe0,0x00,0xf0,0xf3]
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16(<8 x i16>* %A) nounwind {
	%tmp1 = load <8 x i16>* %A
; CHECK: vrev32.16	q8, q8          @ encoding: [0xe0,0x00,0xf4,0xf3]
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i16> %tmp2
}

define <8 x i8> @test_vrev16D8(<8 x i8>* %A) nounwind {
	%tmp1 = load <8 x i8>* %A
; CHECK: vrev16.8	d16, d16        @ encoding: [0x20,0x01,0xf0,0xf3]
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i8> %tmp2
}

define <16 x i8> @test_vrev16Q8(<16 x i8>* %A) nounwind {
	%tmp1 = load <16 x i8>* %A
; CHECK: vrev16.8	q8, q8          @ encoding: [0x60,0x01,0xf0,0xf3]
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
	ret <16 x i8> %tmp2
}
