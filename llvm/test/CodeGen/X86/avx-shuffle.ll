; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; PR11102
define <4 x float> @test1(<4 x float> %a) nounwind {
  %b = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 2, i32 5, i32 undef, i32 undef>
  ret <4 x float> %b
; CHECK: test1:
; CHECK: vshufps
; CHECK: vpshufd
}

; rdar://10538417
define <3 x i64> @test2(<2 x i64> %v) nounwind readnone {
; CHECK: test2:
; CHECK: vxorpd
; CHECK: vmovsd
  %1 = shufflevector <2 x i64> %v, <2 x i64> %v, <3 x i32> <i32 0, i32 1, i32 undef>
  %2 = shufflevector <3 x i64> zeroinitializer, <3 x i64> %1, <3 x i32> <i32 3, i32 4, i32 2>
  ret <3 x i64> %2
}
