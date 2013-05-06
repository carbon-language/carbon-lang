; Test 32-bit floating-point comparison.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparison with registers.
define i64 @f1(i64 %a, i64 %b, float %f1, float %f2) {
; CHECK: f1:
; CHECK: cebr %f0, %f2
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the low end of the CEB range.
define i64 @f2(i64 %a, i64 %b, float %f1, float *%ptr) {
; CHECK: f2:
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the high end of the aligned CEB range.
define i64 @f3(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK: f3:
; CHECK: ceb %f0, 4092(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1023
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK: f4:
; CHECK: aghi %r4, 4096
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 1024
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check negative displacements, which also need separate address logic.
define i64 @f5(i64 %a, i64 %b, float %f1, float *%base) {
; CHECK: f5:
; CHECK: aghi %r4, -4
; CHECK: ceb %f0, 0(%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr float *%base, i64 -1
  %f2 = load float *%ptr
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that CEB allows indices.
define i64 @f6(i64 %a, i64 %b, float %f1, float *%base, i64 %index) {
; CHECK: f6:
; CHECK: sllg %r1, %r5, 2
; CHECK: ceb %f0, 400(%r1,%r4)
; CHECK-NEXT: j{{g?}}e
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %ptr1 = getelementptr float *%base, i64 %index
  %ptr2 = getelementptr float *%ptr1, i64 100
  %f2 = load float *%ptr2
  %cond = fcmp oeq float %f1, %f2
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
