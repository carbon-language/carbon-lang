; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

define i32 @t1(i32 %a, i32 %b) {
; CHECK: @t1
; CHECK: sdiv w2, w0, w1
; CHECK: msub w2, w2, w1, w0
  %1 = srem i32 %a, %b
  ret i32 %1
}

define i64 @t2(i64 %a, i64 %b) {
; CHECK: @t2
; CHECK: sdiv x2, x0, x1
; CHECK: msub x2, x2, x1, x0
  %1 = srem i64 %a, %b
  ret i64 %1
}

define i32 @t3(i32 %a, i32 %b) {
; CHECK: @t3
; CHECK: udiv w2, w0, w1
; CHECK: msub w2, w2, w1, w0
  %1 = urem i32 %a, %b
  ret i32 %1
}

define i64 @t4(i64 %a, i64 %b) {
; CHECK: @t4
; CHECK: udiv x2, x0, x1
; CHECK: msub x2, x2, x1, x0
  %1 = urem i64 %a, %b
  ret i64 %1
}
