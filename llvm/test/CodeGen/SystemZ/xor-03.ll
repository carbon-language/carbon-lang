; Test 64-bit XORs in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check XGR.
define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1:
; CHECK: xgr %r2, %r3
; CHECK: br %r14
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check XG with no displacement.
define i64 @f2(i64 %a, i64 *%src) {
; CHECK: f2:
; CHECK: xg %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64 *%src
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check the high end of the aligned XG range.
define i64 @f3(i64 %a, i64 *%src) {
; CHECK: f3:
; CHECK: xg %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65535
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 *%src) {
; CHECK: f4:
; CHECK: agfi %r3, 524288
; CHECK: xg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 65536
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check the high end of the negative aligned XG range.
define i64 @f5(i64 %a, i64 *%src) {
; CHECK: f5:
; CHECK: xg %r2, -8(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -1
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check the low end of the XG range.
define i64 @f6(i64 %a, i64 *%src) {
; CHECK: f6:
; CHECK: xg %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65536
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, i64 *%src) {
; CHECK: f7:
; CHECK: agfi %r3, -524296
; CHECK: xg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64 *%src, i64 -65537
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}

; Check that XG allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK: f8:
; CHECK: xg %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64 *%ptr
  %xor = xor i64 %a, %b
  ret i64 %xor
}
