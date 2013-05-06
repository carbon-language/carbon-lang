; Test 64-bit comparison in which the second operand is a zero-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check unsigned register comparison.
define double @f1(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f1:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation.
define double @f2(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK: f2:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed register comparison, which can't use CLGFR.
define double @f3(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f3:
; CHECK-NOT: clgfr
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f4(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK: f4:
; CHECK-NOT: clgfr
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register equality.
define double @f5(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f5:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f6(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK: f6:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register inequality.
define double @f7(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK: f7:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}lh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f8(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK: f8:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: j{{g?}}lh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparisonn with memory.
define double @f9(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f9:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison with memory.
define double @f10(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f10:
; CHECK-NOT: clgf
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory equality.
define double @f11(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f11:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}e
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory inequality.
define double @f12(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK: f12:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}lh
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CLGF range.
define double @f13(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f13:
; CHECK: clgf %r2, 524284(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131071
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f14(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f14:
; CHECK: agfi %r3, 524288
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 131072
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CLGF range.
define double @f15(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f15:
; CHECK: clgf %r2, -4(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -1
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CLGF range.
define double @f16(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f16:
; CHECK: clgf %r2, -524288(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131072
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f17(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK: f17:
; CHECK: agfi %r3, -524292
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32 *%base, i64 -131073
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLGF allows an index.
define double @f18(double %a, double %b, i64 %i1, i64 %base, i64 %index) {
; CHECK: f18:
; CHECK: clgf %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: j{{g?}}l
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %unext = load i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
