; Test multiplication of two f32s, producing an f64 result.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register multiplication.
define double @f1(float %f1, float %f2) {
; CHECK: f1:
; CHECK: mdebr %f0, %f2
; CHECK: br %r14
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}

; Check the low end of the MDEB range.
define double @f2(float %f1, float *%ptr) {
; CHECK: f2:
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}

; Check the high end of the aligned MDEB range.
define double @f3(float %f1, float *%base) {
; CHECK: f3:
; CHECK: mdeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %f2 = load float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(float %f1, float *%base) {
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %f2 = load float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(float %f1, float *%base) {
; CHECK: f5:
; CHECK: aghi %r2, -4
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %f2 = load float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}

; Check that MDEB allows indices.
define double @f6(float %f1, float *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: mdeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float *%base, i64 %index
  %ptr2 = getelementptr float *%ptr1, i64 100
  %f2 = load float *%ptr2
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = fmul double %f1x, %f2x
  ret double %res
}
