; Test 64-bit unsigned comparisons in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check a value near the low end of the range.  We use CGFI for comparisons
; with zero, or things that are equivalent to them.
define double @f1(double %a, double %b, i64 %i1) {
; CHECK: f1:
; CHECK: clgfi %r2, 1
; CHECK-NEXT: j{{g?}}h
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ugt i64 %i1, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CLGFI range.
define double @f2(double %a, double %b, i64 %i1) {
; CHECK: f2:
; CHECK: clgfi %r2, 4294967295
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i64 %i1, 4294967295
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which must use a register comparison.
define double @f3(double %a, double %b, i64 %i1) {
; CHECK: f3:
; CHECK: clgr %r2,
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp ult i64 %i1, 4294967296
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
