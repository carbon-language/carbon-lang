; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@x = common global i32 0
@y = common global i32 0

define void @test_01(i32 %N) {
  %1 = alloca i32
  store i32 %N, i32* %1
  store i32 10, i32* @x
  %2 = load i32, i32* %1
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %5

; <label>:4
  store i32 5, i32* @x
  br label %5

; <label>:5
  store i32 15, i32* @x
  ret void
}
; CHECK-LABEL: @test_01(
; CHECK-NOT: store i32 10, i32* @x
; CHECK-NOT: store i32 5, i32* @x
; CHECK: store i32 15, i32* @x


define void @test_02(i32 %N) {
  %1 = alloca i32
  store i32 %N, i32* %1
  store i32 10, i32* @x
  %2 = load i32, i32* %1
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %5

; <label>:4
  store i32 5, i32* @x
  br label %7

; <label>:5
  %6 = load i32, i32* @x
  store i32 %6, i32* @y
  br label %7

; <label>:7
  store i32 15, i32* @x
  ret void
}
; CHECK-LABEL: @test_02(
; CHECK: store i32 10, i32* @x
; CHECK-NOT: store i32 5, i32* @x
; CHECK: store i32 %6, i32* @y


define void @test_03(i32 %N) #0 {
  %1 = alloca i32
  store i32 %N, i32* %1
  store i32 10, i32* @x
  %2 = load i32, i32* %1
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %4, label %6

; <label>:4                                       ; preds = %0
  %5 = load i32, i32* @x
  store i32 %5, i32* @y
  br label %6

; <label>:6                                       ; preds = %4, %0
  store i32 15, i32* @x
  ret void
}
; CHECK-LABEL: @test_03(
; CHECK: store i32 10, i32* @x
; CHECK: store i32 %5, i32* @y
; CHECK: store i32 15, i32* @x
