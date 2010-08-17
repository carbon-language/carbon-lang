; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @test_vrev64D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev64D8:
;CHECK: vrev64.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev64D16(<4 x i16>* %A) nounwind {
;CHECK: test_vrev64D16:
;CHECK: vrev64.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
	ret <4 x i16> %tmp2
}

define <2 x i32> @test_vrev64D32(<2 x i32>* %A) nounwind {
;CHECK: test_vrev64D32:
;CHECK: vrev64.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x i32> %tmp2
}

define <2 x float> @test_vrev64Df(<2 x float>* %A) nounwind {
;CHECK: test_vrev64Df:
;CHECK: vrev64.32
	%tmp1 = load <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x float> %tmp2
}

define <16 x i8> @test_vrev64Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev64Q8:
;CHECK: vrev64.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev64Q16(<8 x i16>* %A) nounwind {
;CHECK: test_vrev64Q16:
;CHECK: vrev64.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i16> %tmp2
}

define <4 x i32> @test_vrev64Q32(<4 x i32>* %A) nounwind {
;CHECK: test_vrev64Q32:
;CHECK: vrev64.32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i32> %tmp2
}

define <4 x float> @test_vrev64Qf(<4 x float>* %A) nounwind {
;CHECK: test_vrev64Qf:
;CHECK: vrev64.32
	%tmp1 = load <4 x float>* %A
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x float> %tmp2
}

define <8 x i8> @test_vrev32D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev32D8:
;CHECK: vrev32.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev32D16(<4 x i16>* %A) nounwind {
;CHECK: test_vrev32D16:
;CHECK: vrev32.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i16> %tmp2
}

define <16 x i8> @test_vrev32Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev32Q8:
;CHECK: vrev32.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16(<8 x i16>* %A) nounwind {
;CHECK: test_vrev32Q16:
;CHECK: vrev32.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i16> %tmp2
}

define <8 x i8> @test_vrev16D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev16D8:
;CHECK: vrev16.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i8> %tmp2
}

define <16 x i8> @test_vrev16Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev16Q8:
;CHECK: vrev16.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
	ret <16 x i8> %tmp2
}

; Undef shuffle indices should not prevent matching to VREV:

define <8 x i8> @test_vrev64D8_undef(<8 x i8>* %A) nounwind {
;CHECK: test_vrev64D8_undef:
;CHECK: vrev64.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 undef, i32 undef, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16_undef(<8 x i16>* %A) nounwind {
;CHECK: test_vrev32Q16_undef:
;CHECK: vrev32.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 undef, i32 0, i32 undef, i32 2, i32 5, i32 4, i32 7, i32 undef>
	ret <8 x i16> %tmp2
}
