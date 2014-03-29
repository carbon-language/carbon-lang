; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

; Materialize using fmov
define void @float_(float* %value) {
; CHECK: @float_
; CHECK: fmov s0, #1.250000e+00
  store float 1.250000e+00, float* %value, align 4
  ret void
}

define void @double_(double* %value) {
; CHECK: @double_
; CHECK: fmov d0, #1.250000e+00
  store double 1.250000e+00, double* %value, align 8
  ret void
}

; Materialize from constant pool
define float @float_cp() {
; CHECK: @float_cp
  ret float 0x400921FB60000000
}

define double @double_cp() {
; CHECK: @double_cp
  ret double 0x400921FB54442D18
}
