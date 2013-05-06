; Test conversions of unsigned i32s to floating-point values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check i32->f32.  There is no native instruction, so we must promote
; to i64 first.
define float @f1(i32 %i) {
; CHECK: f1:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cegbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = uitofp i32 %i to float
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) {
; CHECK: f2:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cdgbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = uitofp i32 %i to double
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, fp128 *%dst) {
; CHECK: f3:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cxgbr %f0, [[REGISTER]]
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = uitofp i32 %i to fp128
  store fp128 %conv, fp128 *%dst
  ret void
}
