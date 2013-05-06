; Test 32-bit signed comparison in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check comparisons with 0.
define double @f1(double %a, double %b, i32 %i1) {
; CHECK: f1:
; CHECK: chi %r2, 0
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 0
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check comparisons with 1.
define double @f2(double %a, double %b, i32 %i1) {
; CHECK: f2:
; CHECK: chi %r2, 1
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the CHI range.
define double @f3(double %a, double %b, i32 %i1) {
; CHECK: f3:
; CHECK: chi %r2, 32767
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 32767
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which must use CFI.
define double @f4(double %a, double %b, i32 %i1) {
; CHECK: f4:
; CHECK: cfi %r2, 32768
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, 32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the signed 32-bit range.
define double @f5(double %a, double %b, i32 %i1) {
; CHECK: f5:
; CHECK: cfi %r2, 2147483647
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, 2147483647
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value up, which should be treated as a negative value.
define double @f6(double %a, double %b, i32 %i1) {
; CHECK: f6:
; CHECK: cfi %r2, -2147483648
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, 2147483648
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative CHI range.
define double @f7(double %a, double %b, i32 %i1) {
; CHECK: f7:
; CHECK: chi %r2, -1
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CHI range.
define double @f8(double %a, double %b, i32 %i1) {
; CHECK: f8:
; CHECK: chi %r2, -32768
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -32768
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, which must use CFI instead.
define double @f9(double %a, double %b, i32 %i1) {
; CHECK: f9:
; CHECK: cfi %r2, -32769
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, -32769
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the signed 32-bit range.
define double @f10(double %a, double %b, i32 %i1) {
; CHECK: f10:
; CHECK: cfi %r2, -2147483648
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, -2147483648
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next value down, which should be treated as a positive value.
define double @f11(double %a, double %b, i32 %i1) {
; CHECK: f11:
; CHECK: cfi %r2, 2147483647
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp eq i32 %i1, -2147483649
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
