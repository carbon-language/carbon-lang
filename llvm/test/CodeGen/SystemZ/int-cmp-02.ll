; Test 32-bit signed comparison in which the second operand is a variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register comparison.
define double @f1(double %a, double %b, i32 %i1, i32 %i2) {
; CHECK: f1:
; CHECK: cr %r2, %r3
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the C range.
define double @f2(double %a, double %b, i32 %i1, i32 *%ptr) {
; CHECK: f2:
; CHECK: c %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned C range.
define double @f3(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f3:
; CHECK: c %r2, 4092(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 1023
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which should use CY instead of C.
define double @f4(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f4:
; CHECK: cy %r2, 4096(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 1024
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CY range.
define double @f5(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f5:
; CHECK: cy %r2, 524284(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131071
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f6(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f6:
; CHECK: agfi %r3, 524288
; CHECK: c %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131072
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CY range.
define double @f7(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f7:
; CHECK: cy %r2, -4(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -1
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CY range.
define double @f8(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f8:
; CHECK: cy %r2, -524288(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131072
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f9(double %a, double %b, i32 %i1, i32 *%base) {
; CHECK: f9:
; CHECK: agfi %r3, -524292
; CHECK: c %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131073
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that C allows an index.
define double @f10(double %a, double %b, i32 %i1, i64 %base, i64 %index) {
; CHECK: f10:
; CHECK: c %r2, 4092({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to i32 *
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CY allows an index.
define double @f11(double %a, double %b, i32 %i1, i64 %base, i64 %index) {
; CHECK: f11:
; CHECK: cy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i32 *
  %i2 = load i32 *%ptr
  %cond = icmp slt i32 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
