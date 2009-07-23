; RUN: llvm-as < %s | opt -instcombine | llvm-dis | FileCheck %s

; This turns into a&1 != 0
define <2 x i1> @test1(<2 x i64> %a) {
  %t = trunc <2 x i64> %a to <2 x i1>
  ret <2 x i1> %t

; CHECK: define <2 x i1> @test1
; CHECK:   and <2 x i64> %a, <i64 1, i64 1>
; CHECK:   icmp ne <2 x i64> %tmp, zeroinitializer
}

; The ashr turns into an lshr.
define <2 x i64> @test2(<2 x i64> %a) {
  %b = and <2 x i64> %a, <i64 65535, i64 65535>
  %t = ashr <2 x i64> %b, <i64 1, i64 1>
  ret <2 x i64> %t

; CHECK: define <2 x i64> @test2
; CHECK:   and <2 x i64> %a, <i64 65535, i64 65535>
; CHECK:   lshr <2 x i64> %b, <i64 1, i64 1>
}
