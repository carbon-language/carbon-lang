; Test that the abs library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare i32 @abs(i32)
declare i64 @labs(i64)
declare i64 @llabs(i64)

; Check abs(x) -> x >s -1 ? x : -x.

define i32 @test_simplify1(i32 %x) {
; CHECK-LABEL: @test_simplify1(
  %ret = call i32 @abs(i32 %x)
; CHECK-NEXT: [[ISPOS:%[a-z0-9]+]] = icmp sgt i32 %x, -1
; CHECK-NEXT: [[NEG:%[a-z0-9]+]] = sub i32 0, %x
; CHECK-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[ISPOS]], i32 %x, i32 [[NEG]]
  ret i32 %ret
; CHECK-NEXT: ret i32 [[RET]]
}

define i64 @test_simplify2(i64 %x) {
; CHECK-LABEL: @test_simplify2(
  %ret = call i64 @labs(i64 %x)
; CHECK-NEXT: [[ISPOS:%[a-z0-9]+]] = icmp sgt i64 %x, -1
; CHECK-NEXT: [[NEG:%[a-z0-9]+]] = sub i64 0, %x
; CHECK-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[ISPOS]], i64 %x, i64 [[NEG]]
  ret i64 %ret
; CHECK-NEXT: ret i64 [[RET]]
}

define i64 @test_simplify3(i64 %x) {
; CHECK-LABEL: @test_simplify3(
  %ret = call i64 @llabs(i64 %x)
; CHECK-NEXT: [[ISPOS:%[a-z0-9]+]] = icmp sgt i64 %x, -1
; CHECK-NEXT: [[NEG:%[a-z0-9]+]] = sub i64 0, %x
; CHECK-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[ISPOS]], i64 %x, i64 [[NEG]]
  ret i64 %ret
; CHECK-NEXT: ret i64 [[RET]]
}
