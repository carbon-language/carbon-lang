; RUN: llc < %s -march=x86 -mcpu=core2 | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=yonah | FileCheck --check-prefix=YONAH %s

define <4 x i32> @test1(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: pshufd
; CHECK-YONAH: pshufd
  %C = shufflevector <4 x i32> %A, <4 x i32> undef, <4 x i32> < i32 1, i32 2, i32 3, i32 0 >
	ret <4 x i32> %C
}

define <4 x i32> @test2(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: palignr
; CHECK-YONAH: shufps
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 1, i32 2, i32 3, i32 4 >
	ret <4 x i32> %C
}

define <4 x i32> @test3(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 1, i32 2, i32 undef, i32 4 >
	ret <4 x i32> %C
}

define <4 x i32> @test4(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> < i32 6, i32 7, i32 undef, i32 1 >
	ret <4 x i32> %C
}

define <4 x float> @test5(<4 x float> %A, <4 x float> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 6, i32 7, i32 undef, i32 1 >
	ret <4 x float> %C
}

define <8 x i16> @test6(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 3, i32 4, i32 undef, i32 6, i32 7, i32 8, i32 9, i32 10 >
	ret <8 x i16> %C
}

define <8 x i16> @test7(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 undef, i32 6, i32 undef, i32 8, i32 9, i32 10, i32 11, i32 12 >
	ret <8 x i16> %C
}

define <8 x i16> @test8(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 undef, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0 >
	ret <8 x i16> %C
}

define <16 x i8> @test9(<16 x i8> %A, <16 x i8> %B) nounwind {
; CHECK: palignr
  %C = shufflevector <16 x i8> %A, <16 x i8> %B, <16 x i32> < i32 5, i32 6, i32 7, i32 undef, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20 >
	ret <16 x i8> %C
}
