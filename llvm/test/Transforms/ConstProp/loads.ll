; RUN: opt < %s -instcombine -S | FileCheck %s 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@g1 = constant {{i32,i8},i32} {{i32,i8} { i32 -559038737, i8 186 }, i32 -889275714 }
@g2 = constant double 1.0
@g3 = constant {i64, i64} { i64 123, i64 112312312 }

; Simple load
define i32 @test1() {
  %r = load i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0)
  ret i32 %r
; CHECK: @test1
; CHECK: ret i32 -559038737
}

; PR3152
; Load of first 16 bits of 32-bit value.
define i16 @test2() {
  %r = load i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*)
  ret i16 %r

; CHECK: @test2
; CHECK: ret i16 -16657 
}

; Load of second 16 bits of 32-bit value.
define i16 @test3() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 1)
  ret i16 %r

; CHECK: @test3
; CHECK: ret i16 -8531
}

; Load of 8 bit field + tail padding.
define i16 @test4() {
  %r = load i16* getelementptr(i16* bitcast(i32* getelementptr ({{i32,i8},i32}* @g1, i32 0, i32 0, i32 0) to i16*), i32 2)
  ret i16 %r
; CHECK: @test4
; CHECK: ret i16 186
}

; Load of double bits.
define i64 @test6() {
  %r = load i64* bitcast(double* @g2 to i64*)
  ret i64 %r

; CHECK: @test6
; CHECK: ret i64 4607182418800017408
}

; Load of double bits.
define i16 @test7() {
  %r = load i16* bitcast(double* @g2 to i16*)
  ret i16 %r

; CHECK: @test7
; CHECK: ret i16 0
}

; Double load.
define double @test8() {
  %r = load double* bitcast({{i32,i8},i32}* @g1 to double*)
  ret double %r

; CHECK: @test8
; CHECK: ret double 0xBADEADBEEF
}


; i128 load.
define i128 @test9() {
  %r = load i128* bitcast({i64, i64}* @g3 to i128*)
  ret i128 %r

; CHECK: @test9
; CHECK: ret i128 2071796475790618158476296315
}

; vector load.
define <2 x i64> @test10() {
  %r = load <2 x i64>* bitcast({i64, i64}* @g3 to <2 x i64>*)
  ret <2 x i64> %r

; CHECK: @test10
; CHECK: ret <2 x i64> <i64 123, i64 112312312>
}


; PR5287
@g4 = internal constant { i8, i8 } { i8 -95, i8 8 }

define i16 @test11() nounwind {
entry:
  %a = load i16* bitcast ({ i8, i8 }* @g4 to i16*)
  ret i16 %a
  
; CHECK: @test11
; CHECK: ret i16 2209
}


; PR5551
@test12g = private constant [6 x i8] c"a\00b\00\00\00"

define i16 @test12() {
  %a = load i16* getelementptr inbounds ([3 x i16]* bitcast ([6 x i8]* @test12g to [3 x i16]*), i32 0, i64 1) 
  ret i16 %a
; CHECK: @test12
; CHECK: ret i16 98
}


; PR5978
@g5 = constant i8 4
define i1 @test13() {
  %A = load i1* bitcast (i8* @g5 to i1*)
  ret i1 %A
; CHECK: @test13
; CHECK: ret i1 false
}
