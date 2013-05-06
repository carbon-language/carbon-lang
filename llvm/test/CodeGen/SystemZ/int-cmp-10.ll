; Test 32-bit unsigned comparisons in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a value near the low end of the range.  We use CFI for comparisons
; with zero, or things that are equivalent to them.
define double @f1(double %a, double %b, i32 %i1) {
; CHECK: f1:
; CHECK: clfi %r2, 1
; CHECK-NEXT: j{{g?}}h
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ugt i32 %i1, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check a value near the high end of the range.
define double @f2(double %a, double %b, i32 %i1) {
; CHECK: f2:
; CHECK: clfi %r2, 4294967280
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i32 %i1, 4294967280
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
