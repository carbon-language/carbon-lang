; Test multiplication of two f64s, producing an f128 result.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register multiplication.  "mxdbr %f0, %f2" is not valid from LLVM's
; point of view, because %f2 is the low register of the FP128 %f0.  Pass the
; multiplier in %f4 instead.
define void @f1(double %f1, double %dummy, double %f2, fp128 *%dst) {
; CHECK: f1:
; CHECK: mxdbr %f0, %f4
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the low end of the MXDB range.
define void @f2(double %f1, double *%ptr, fp128 *%dst) {
; CHECK: f2:
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %f2 = load double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the high end of the aligned MXDB range.
define void @f3(double %f1, double *%base, fp128 *%dst) {
; CHECK: f3:
; CHECK: mxdb %f0, 4088(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 511
  %f2 = load double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(double %f1, double *%base, fp128 *%dst) {
; CHECK: f4:
; CHECK: aghi %r2, 4096
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 512
  %f2 = load double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(double %f1, double *%base, fp128 *%dst) {
; CHECK: f5:
; CHECK: aghi %r2, -8
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 -1
  %f2 = load double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that MXDB allows indices.
define void @f6(double %f1, double *%base, i64 %index, fp128 *%dst) {
; CHECK: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: mxdb %f0, 800(%r1,%r2)
; CHECK: std %f0, 0(%r4)
; CHECK: std %f2, 8(%r4)
; CHECK: br %r14
  %ptr1 = getelementptr double *%base, i64 %index
  %ptr2 = getelementptr double *%ptr1, i64 100
  %f2 = load double *%ptr2
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, fp128 *%dst
  ret void
}
