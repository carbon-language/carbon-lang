; Test that the ffs* library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; RUN: opt < %s -mtriple i386-pc-linux -instcombine -S | FileCheck %s -check-prefix=CHECK-FFS
; RUN: opt -instcombine -mtriple=arm64-apple-ios9.0 -S %s | FileCheck --check-prefix=CHECK-FFS %s
; RUN: opt -instcombine -mtriple=arm64-apple-tvos9.0 -S %s | FileCheck --check-prefix=CHECK-FFS %s
; RUN: opt -instcombine -mtriple=thumbv7k-apple-watchos2.0 -S %s | FileCheck --check-prefix=CHECK-FFS %s
; RUN: opt -instcombine -mtriple=x86_64-apple-macosx10.11 -S %s | FileCheck --check-prefix=CHECK-FFS %s
; RUN: opt -instcombine -mtriple=x86_64-freebsd-gnu -S %s | FileCheck --check-prefix=CHECK-FFS %s

declare i32 @ffs(i32)
declare i32 @ffsl(i32)
declare i32 @ffsll(i64)

; Check ffs(0) -> 0.

define i32 @test_simplify1() {
; CHECK-LABEL: @test_simplify1(
  %ret = call i32 @ffs(i32 0)
  ret i32 %ret
; CHECK-NEXT: ret i32 0
}

define i32 @test_simplify2() {
; CHECK-FFS-LABEL: @test_simplify2(
  %ret = call i32 @ffsl(i32 0)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 0
}

define i32 @test_simplify3() {
; CHECK-FFS-LABEL: @test_simplify3(
  %ret = call i32 @ffsll(i64 0)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 0
}

; Check ffs(c) -> cttz(c) + 1, where 'c' is a constant.

define i32 @test_simplify4() {
; CHECK-LABEL: @test_simplify4(
  %ret = call i32 @ffs(i32 1)
  ret i32 %ret
; CHECK-NEXT: ret i32 1
}

define i32 @test_simplify5() {
; CHECK-LABEL: @test_simplify5(
  %ret = call i32 @ffs(i32 2048)
  ret i32 %ret
; CHECK-NEXT: ret i32 12
}

define i32 @test_simplify6() {
; CHECK-LABEL: @test_simplify6(
  %ret = call i32 @ffs(i32 65536)
  ret i32 %ret
; CHECK-NEXT: ret i32 17
}

define i32 @test_simplify7() {
; CHECK-FFS-LABEL: @test_simplify7(
  %ret = call i32 @ffsl(i32 65536)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 17
}

define i32 @test_simplify8() {
; CHECK-FFS-LABEL: @test_simplify8(
  %ret = call i32 @ffsll(i64 1024)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 11
}

define i32 @test_simplify9() {
; CHECK-FFS-LABEL: @test_simplify9(
  %ret = call i32 @ffsll(i64 65536)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 17
}

define i32 @test_simplify10() {
; CHECK-FFS-LABEL: @test_simplify10(
  %ret = call i32 @ffsll(i64 17179869184)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 35
}

define i32 @test_simplify11() {
; CHECK-FFS-LABEL: @test_simplify11(
  %ret = call i32 @ffsll(i64 281474976710656)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 49
}

define i32 @test_simplify12() {
; CHECK-FFS-LABEL: @test_simplify12(
  %ret = call i32 @ffsll(i64 1152921504606846976)
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 61
}

; Check ffs(x) -> x != 0 ? (i32)llvm.cttz(x) + 1 : 0.

define i32 @test_simplify13(i32 %x) {
; CHECK-LABEL: @test_simplify13(
  %ret = call i32 @ffs(i32 %x)
; CHECK-NEXT: [[CTTZ:%[a-z0-9]+]] = call i32 @llvm.cttz.i32(i32 %x, i1 true)
; CHECK-NEXT: [[INC:%[a-z0-9]+]] = add nuw nsw i32 [[CTTZ]], 1
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 %x, 0
; CHECK-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[CMP]], i32 [[INC]], i32 0
  ret i32 %ret
; CHECK-NEXT: ret i32 [[RET]]
}

define i32 @test_simplify14(i32 %x) {
; CHECK-FFS-LABEL: @test_simplify14(
  %ret = call i32 @ffsl(i32 %x)
; CHECK-FFS-NEXT: [[CTTZ:%[a-z0-9]+]] = call i32 @llvm.cttz.i32(i32 %x, i1 true)
; CHECK-FFS-NEXT: [[INC:%[a-z0-9]+]] = add nuw nsw i32 [[CTTZ]], 1
; CHECK-FFS-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 %x, 0
; CHECK-FFS-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[CMP]], i32 [[INC]], i32 0
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 [[RET]]
}

define i32 @test_simplify15(i64 %x) {
; CHECK-FFS-LABEL: @test_simplify15(
  %ret = call i32 @ffsll(i64 %x)
; CHECK-FFS-NEXT: [[CTTZ:%[a-z0-9]+]] = call i64 @llvm.cttz.i64(i64 %x, i1 true)
; CHECK-FFS-NEXT: [[INC:%[a-z0-9]+]] = add nuw nsw i64 [[CTTZ]], 1
; CHECK-FFS-NEXT: [[TRUNC:%[a-z0-9]+]] = trunc i64 [[INC]] to i32
; CHECK-FFS-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i64 %x, 0
; CHECK-FFS-NEXT: [[RET:%[a-z0-9]+]] = select i1 [[CMP]], i32 [[TRUNC]], i32 0
  ret i32 %ret
; CHECK-FFS-NEXT: ret i32 [[RET]]
}
