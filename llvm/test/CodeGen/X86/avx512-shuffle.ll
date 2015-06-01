; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: vpermps
; CHECK: ret
define <16 x float> @test1(<16 x float> %a) nounwind {
  %c = shufflevector <16 x float> %a, <16 x float> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x float> %c
}

; CHECK-LABEL: test2:
; CHECK: vpermd
; CHECK: ret
define <16 x i32> @test2(<16 x i32> %a) nounwind {
  %c = shufflevector <16 x i32> %a, <16 x i32> undef, <16 x i32> <i32 2, i32 5, i32 undef, i32 undef, i32 7, i32 undef, i32 10, i32 1,  i32 0, i32 5, i32 undef, i32 4, i32 7, i32 undef, i32 10, i32 1>
  ret <16 x i32> %c
}

; CHECK-LABEL: test3:
; CHECK: vpermq
; CHECK: ret
define <8 x i64> @test3(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 5, i32 1, i32 undef, i32 7, i32 undef, i32 3, i32 1>
  ret <8 x i64> %c
}

; CHECK-LABEL: test4:
; CHECK: vpermpd
; CHECK: ret
define <8 x double> @test4(<8 x double> %a) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> undef, <8 x i32> <i32 1, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x double> %c
}

; CHECK-LABEL: test5:
; CHECK: vpermt2pd
; CHECK: ret
define <8 x double> @test5(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 8, i32 0, i32 1, i32 6, i32 10, i32 4, i32 5>
  ret <8 x double> %c
}

; CHECK-LABEL: test6:
; CHECK: vpermq $30
; CHECK: ret
define <8 x i64> @test6(<8 x i64> %a) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> undef, <8 x i32> <i32 2, i32 3, i32 1, i32 0, i32 6, i32 7, i32 5, i32 4>
  ret <8 x i64> %c
}

; CHECK-LABEL: test7:
; CHECK: vpermt2q
; CHECK: ret
define <8 x i64> @test7(<8 x i64> %a, <8 x i64> %b) nounwind {
  %c = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 8, i32 0, i32 1, i32 6, i32 10, i32 4, i32 5>
  ret <8 x i64> %c
}

; CHECK-LABEL: test14
; CHECK: vpermilpd $203, %zmm
; CHECK: ret
define <8 x double> @test14(<8 x double> %a) {
 %b = shufflevector <8 x double> %a, <8 x double> undef, <8 x i32><i32 1, i32 1, i32 2, i32 3, i32 4, i32 4, i32 7, i32 7>
 ret <8 x double> %b
}

; CHECK-LABEL: test17
; CHECK: vshufpd $19, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test17(<8 x double> %a, <8 x double> %b) nounwind {
  %c = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 9, i32 2, i32 10, i32 5, i32 undef, i32 undef, i32 undef>
  ret <8 x double> %c
}

; CHECK-LABEL: test20
; CHECK: vpunpckhqdq  %zmm
; CHECK: ret
define <8 x i64> @test20(<8 x i64> %a, <8 x i64> %c) {
 %b = shufflevector <8 x i64> %a, <8 x i64> %c, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
 ret <8 x i64> %b
}

; CHECK-LABEL: test21
; CHECK: vbroadcastsd  %xmm0, %zmm
; CHECK: ret
define <8 x double> @test21(<8 x double> %a, <8 x double> %b) {
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

; CHECK-LABEL: test22
; CHECK: vpbroadcastq  %xmm0, %zmm
; CHECK: ret
define <8 x i64> @test22(<8 x i64> %a, <8 x i64> %b) {
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

