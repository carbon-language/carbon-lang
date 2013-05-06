; Test 64-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare double @llvm.sqrt.f64(double %f)

; Check register square root.
define double @f1(double %val) {
; CHECK: f1:
; CHECK: sqdbr %f0, %f0
; CHECK: br %r14
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the low end of the SQDB range.
define double @f2(double *%ptr) {
; CHECK: f2:
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %val = load double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the high end of the aligned SQDB range.
define double @f3(double *%base) {
; CHECK: f3:
; CHECK: sqdb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 511
  %val = load double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double *%base) {
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 512
  %val = load double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double *%base) {
; CHECK: f5:
; CHECK: aghi %r2, -8
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 -1
  %val = load double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check that SQDB allows indices.
define double @f6(double *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: sqdb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double *%base, i64 %index
  %ptr2 = getelementptr double *%ptr1, i64 100
  %val = load double *%ptr2
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}
