; RUN: llvm-as < %s | llc -march=arm -mattr=+neon | FileCheck %s

define arm_apcscc <8 x i8> @test_vextd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: test_vextd:
;CHECK: vext
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
	ret <8 x i8> %tmp3
}

define arm_apcscc <8 x i8> @test_vextRd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: test_vextRd:
;CHECK: vext
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4>
	ret <8 x i8> %tmp3
}

define arm_apcscc <16 x i8> @test_vextq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: test_vextq:
;CHECK: vext
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
	ret <16 x i8> %tmp3
}

define arm_apcscc <16 x i8> @test_vextRq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: test_vextRq:
;CHECK: vext
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
	ret <16 x i8> %tmp3
}

define arm_apcscc <4 x i16> @test_vextd16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: test_vextd16:
;CHECK: vext
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i16> %tmp3
}

define arm_apcscc <4 x i32> @test_vextq32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: test_vextq32:
;CHECK: vext
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i32> %tmp3
}

