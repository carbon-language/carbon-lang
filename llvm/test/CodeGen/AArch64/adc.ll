; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

define i128 @test_simple(i128 %a, i128 %b, i128 %c) {
; CHECK-LABEL: test_simple:

  %valadd = add i128 %a, %b
; CHECK: adds [[ADDLO:x[0-9]+]], x0, x2
; CHECK-NEXT: adcs [[ADDHI:x[0-9]+]], x1, x3

  %valsub = sub i128 %valadd, %c
; CHECK: subs x0, [[ADDLO]], x4
; CHECK: sbcs x1, [[ADDHI]], x5

  ret i128 %valsub
; CHECK: ret
}

define i128 @test_imm(i128 %a) {
; CHECK-LABEL: test_imm:

  %val = add i128 %a, 12
; CHECK: adds x0, x0, #12
; CHECK: adcs x1, x1, {{x[0-9]|xzr}}

  ret i128 %val
; CHECK: ret
}

define i128 @test_shifted(i128 %a, i128 %b) {
; CHECK-LABEL: test_shifted:

  %rhs = shl i128 %b, 45

  %val = add i128 %a, %rhs
; CHECK: adds x0, x0, x2, lsl #45
; CHECK: adcs x1, x1, {{x[0-9]}}

  ret i128 %val
; CHECK: ret
}

define i128 @test_extended(i128 %a, i16 %b) {
; CHECK-LABEL: test_extended:

  %ext = sext i16 %b to i128
  %rhs = shl i128 %ext, 3

  %val = add i128 %a, %rhs
; CHECK: adds x0, x0, w2, sxth #3
; CHECK: adcs x1, x1, {{x[0-9]}}

  ret i128 %val
; CHECK: ret
}
