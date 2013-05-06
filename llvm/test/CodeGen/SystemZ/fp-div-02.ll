; Test 64-bit floating-point division.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register division.
define double @f1(double %f1, double %f2) {
; CHECK: f1:
; CHECK: ddbr %f0, %f2
; CHECK: br %r14
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the low end of the DDB range.
define double @f2(double %f1, double *%ptr) {
; CHECK: f2:
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the high end of the aligned DDB range.
define double @f3(double %f1, double *%base) {
; CHECK: f3:
; CHECK: ddb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 511
  %f2 = load double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, double *%base) {
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 512
  %f2 = load double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, double *%base) {
; CHECK: f5:
; CHECK: aghi %r2, -8
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 -1
  %f2 = load double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check that DDB allows indices.
define double @f6(double %f1, double *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: ddb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double *%base, i64 %index
  %ptr2 = getelementptr double *%ptr1, i64 100
  %f2 = load double *%ptr2
  %res = fdiv double %f1, %f2
  ret double %res
}
