; RUN: opt < %s -instcombine -S | FileCheck %s 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@test1 = constant {{i32,i8},i32} {{i32,i8} { i32 -559038737, i8 186 }, i32 -889275714 }
@test2 = constant double 1.0

; Simple load
define i32 @test1() {
  %r = load i32* getelementptr ({{i32,i8},i32}* @test1, i32 0, i32 0, i32 0)
  ret i32 %r
; @test1
; CHECK: ret i32 -559038737
}

; PR3152
; Load of first 16 bits of 32-bit value.
define i16 @test2() {
  %r = load i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @test1, i32 0, i32 0, i32 0) to i16*)
  ret i16 %r

; @test2
; CHECK: ret i16 -16657 
}

; Load of second 16 bits of 32-bit value.
define i16 @test3() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @test1, i32 0, i32 0, i32 0) to i16*), i32 1)
  ret i16 %r

; @test3
; CHECK: ret i16 -8531
}

; Load of 8 bit field + tail padding.
define i16 @test4() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @test1, i32 0, i32 0, i32 0) to i16*), i32 2)
  ret i16 %r
; @test4
; CHECK: ret i16 186
}

; Load of double bits.
define i64 @test6() {
  %r = load i64* bitcast(double* @test2 to i64*)
  ret i64 %r

; @test6
; CHECK: ret i64 4607182418800017408
}

; Load of double bits.
define i16 @test7() {
  %r = load i16* bitcast(double* @test2 to i16*)
  ret i16 %r

; @test7
; CHECK: ret i16 0
}

; Double load.
define double @test8() {
  %r = load double* bitcast({{i32,i8},i32}* @test1 to double*)
  ret double %r

; @test8
; CHECK: ret double 0xDEADBEBA
}
