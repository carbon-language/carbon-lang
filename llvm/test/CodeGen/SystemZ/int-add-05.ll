; Test 64-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check AGR.
define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1:
; CHECK: agr %r2, %r3
; CHECK: br %r14
  %add = add i64 %a, %b
  ret i64 %add
}

; Check AG with no displacement.
define i64 @f2(i64 %a, i64 *%src) {
; CHECK: f2:
; CHECK: ag %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64 *%src
  %add = add i64 %a, %b
  ret i64 %add
}

; Check the high end of the aligned AG range.
define i64 @f3(i64 %a, i64 *%src) {
; CHECK: f3:
; CHECK: ag %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65535
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 *%src) {
; CHECK: f4:
; CHECK: agfi %r3, 524288
; CHECK: ag %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65536
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}

; Check the high end of the negative aligned AG range.
define i64 @f5(i64 %a, i64 *%src) {
; CHECK: f5:
; CHECK: ag %r2, -8(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -1
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}

; Check the low end of the AG range.
define i64 @f6(i64 %a, i64 *%src) {
; CHECK: f6:
; CHECK: ag %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65536
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, i64 *%src) {
; CHECK: f7:
; CHECK: agfi %r3, -524296
; CHECK: ag %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65537
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}

; Check that AG allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK: f8:
; CHECK: ag %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64 *%ptr
  %add = add i64 %a, %b
  ret i64 %add
}
