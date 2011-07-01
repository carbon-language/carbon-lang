; RUN: opt -instsimplify -S < %s | FileCheck %s

; @test0
; CHECK: ret i64 undef
define i64 @test0() {
  %r = mul i64 undef, undef
  ret i64 %r
}

; @test1
; CHECK: ret i64 undef
define i64 @test1() {
  %r = mul i64 3, undef
  ret i64 %r
}

; @test2
; CHECK: ret i64 undef
define i64 @test2() {
  %r = mul i64 undef, 3
  ret i64 %r
}

; @test3
; CHECK: ret i64 0
define i64 @test3() {
  %r = mul i64 undef, 6
  ret i64 %r
}

; @test4
; CHECK: ret i64 0
define i64 @test4() {
  %r = mul i64 6, undef
  ret i64 %r
}

; @test5
; CHECK: ret i64 undef
define i64 @test5() {
  %r = and i64 undef, undef
  ret i64 %r
}

; @test6
; CHECK: ret i64 undef
define i64 @test6() {
  %r = or i64 undef, undef
  ret i64 %r
}

; @test7
; CHECK: ret i64 undef
define i64 @test7() {
  %r = udiv i64 undef, 1
  ret i64 %r
}

; @test8
; CHECK: ret i64 undef
define i64 @test8() {
  %r = sdiv i64 undef, 1
  ret i64 %r
}

; @test9
; CHECK: ret i64 0
define i64 @test9() {
  %r = urem i64 undef, 1
  ret i64 %r
}

; @test10
; CHECK: ret i64 0
define i64 @test10() {
  %r = srem i64 undef, 1
  ret i64 %r
}

; @test11
; CHECK: ret i64 undef
define i64 @test11() {
  %r = shl i64 undef, undef
  ret i64 %r
}

; @test12
; CHECK: ret i64 undef
define i64 @test12() {
  %r = ashr i64 undef, undef
  ret i64 %r
}

; @test13
; CHECK: ret i64 undef
define i64 @test13() {
  %r = lshr i64 undef, undef
  ret i64 %r
}
