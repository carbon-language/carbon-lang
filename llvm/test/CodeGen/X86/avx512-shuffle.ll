; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s
; CHECK: LCP
; CHECK: .long 2
; CHECK: .long 5
; CHECK: .long 0
; CHECK: .long 0
; CHECK: .long 7
; CHECK: .long 0
; CHECK: .long 10
; CHECK: .long 1
; CHECK: .long 0
; CHECK: .long 5
; CHECK: .long 0
; CHECK: .long 4
; CHECK: .long 7
; CHECK: .long 0
; CHECK: .long 10
; CHECK: .long 1
; CHECK: test1:
; CHECK: vpermps
; CHECK: ret
define <16 x float> @test1(<16 x float> %a) nounwind {
  %c = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x float> %c
}

; CHECK: test2:
; CHECK: vpermd
; CHECK: ret
define <16 x i32> @test2(<16 x i32> %a) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x i32> %c
}

; CHECK: test2b:
; CHECK: vpermd
; CHECK: ret
define <16 x i32> @test2b(<16 x i32> %a) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i32> %c
}

; CHECK: test3:
; CHECK: vpermq
; CHECK: ret
define <8 x i64> @test3(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 5, i32 1, i32 undef, i32 7, i32 undef, i32 3, i32 1>
  ret <8 x i64> %c
}

; CHECK: test4:
; CHECK: vpermpd
; CHECK: ret
define <8 x double> @test4(<8 x double> %a) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> undef, <8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x double> %c
}

; CHECK: test5:
; CHECK: vpermi2pd
; CHECK: ret
define <8 x double> @test5(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 8, i32 0, i32 1, i32 6, i32 10, i32 4, i32 5>
  ret <8 x double> %c
}

; CHECK: test6:
; CHECK: vpermq $30
; CHECK: ret
define <8 x i64> @test6(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 1, i32 0, i32 6, i32 7, i32 5, i32 4>
  ret <8 x i64> %c
}

