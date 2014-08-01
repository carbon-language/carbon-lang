; RUN: llc -fast-isel -fast-isel-abort -mtriple=arm64-apple-darwin < %s | FileCheck %s

; Test simple constant offset.
define i64 @test_load1(i64 %a) {
; CHECK-LABEL: test_load1
; CHECK: ldr  x0, [x0, #16]
  %1 = add i64 %a, 16
  %2 = inttoptr i64 %1 to i64*
  %3 = load i64* %2
  ret i64 %3
}

; Test large constant offset.
define i64 @test_load2(i64 %a) {
; CHECK-LABEL: test_load2
; CHECK: add [[REG:x[0-9]+]], x0, {{x[0-9]+}}
; CHECK: ldr  x0, {{\[}}[[REG]]{{\]}}
  %1 = add i64 %a, 16777216
  %2 = inttoptr i64 %1 to i64*
  %3 = load i64* %2
  ret i64 %3
}

