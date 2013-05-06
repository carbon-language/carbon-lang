; Test 64-bit floating-point comparison.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparison with registers.
define i64 @f1(i64 %a, i64 %b, double %f1, double %f2) {
; CHECK: f1:
; CHECK: cdbr %f0, %f2
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the low end of the CDB range.
define i64 @f2(i64 %a, i64 %b, double %f1, double *%ptr) {
; CHECK: f2:
; CHECK: cdb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2 = load double *%ptr
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the high end of the aligned CDB range.
define i64 @f3(i64 %a, i64 %b, double %f1, double *%base) {
; CHECK: f3:
; CHECK: cdb %f0, 4088(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 511
  %f2 = load double *%ptr
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 %b, double %f1, double *%base) {
; CHECK: f4:
; CHECK: aghi %r4, 4096
; CHECK: cdb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 512
  %f2 = load double *%ptr
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check negative displacements, which also need separate address logic.
define i64 @f5(i64 %a, i64 %b, double %f1, double *%base) {
; CHECK: f5:
; CHECK: aghi %r4, -8
; CHECK: cdb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double *%base, i64 -1
  %f2 = load double *%ptr
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that CDB allows indices.
define i64 @f6(i64 %a, i64 %b, double %f1, double *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r5, 3
; CHECK: cdb %f0, 800(%r1,%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr1 = getelementptr double *%base, i64 %index
  %ptr2 = getelementptr double *%ptr1, i64 100
  %f2 = load double *%ptr2
  %cond = fcmp oeq double %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
