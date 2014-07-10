; RUN: llc -fast-isel -fast-isel-abort -mtriple=aarch64 -o - %s | FileCheck %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define void @test_mul8(i8 %lhs, i8 %rhs) {
; CHECK-LABEL: test_mul8:
; CHECK: mul w0, w0, w1
;  %lhs = load i8* @var8
;  %rhs = load i8* @var8
  %prod = mul i8 %lhs, %rhs
  store i8 %prod, i8* @var8
  ret void
}

define void @test_mul16(i16 %lhs, i16 %rhs) {
; CHECK-LABEL: test_mul16:
; CHECK: mul w0, w0, w1
  %prod = mul i16 %lhs, %rhs
  store i16 %prod, i16* @var16
  ret void
}

define void @test_mul32(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_mul32:
; CHECK: mul w0, w0, w1
  %prod = mul i32 %lhs, %rhs
  store i32 %prod, i32* @var32
  ret void
}

define void @test_mul64(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_mul64:
; CHECK: mul x0, x0, x1
  %prod = mul i64 %lhs, %rhs
  store i64 %prod, i64* @var64
  ret void
}
