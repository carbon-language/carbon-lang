; Test 64-bit comparison in which the second operand is a sign-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check signed register comparison.
define double @f1(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f1:
; CHECK: cgfr %r2, %r3
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned register comparison, which can't use CGFR.
define double @f2(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f2:
; CHECK-NOT: cgfr
; CHECK: br %r14
  %i2 = sext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register equality.
define double @f3(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f3:
; CHECK: cgfr %r2, %r3
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = sext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register inequality.
define double @f4(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f4:
; CHECK: cgfr %r2, %r3
; CHECK-NEXT: j{{g?}}lh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = sext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparisonn with memory.
define double @f5(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f5:
; CHECK: cgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison with memory.
define double @f6(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f6:
; CHECK-NOT: cgf
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory equality.
define double @f7(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f7:
; CHECK: cgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory inequality.
define double @f8(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f8:
; CHECK: cgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}lh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CGF range.
define double @f9(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f9:
; CHECK: cgf %r2, 524284(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131071
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f10(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f10:
; CHECK: agfi %r3, 524288
; CHECK: cgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131072
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CGF range.
define double @f11(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f11:
; CHECK: cgf %r2, -4(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -1
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CGF range.
define double @f12(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f12:
; CHECK: cgf %r2, -524288(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131072
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f13(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f13:
; CHECK: agfi %r3, -524292
; CHECK: cgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131073
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CGF allows an index.
define double @f14(double %a, double %b, i64 %i1, i64 %base, i64 %index) {
; CHECK: f14:
; CHECK: cgf %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %unext = load i32 *%ptr
  %i2 = sext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
