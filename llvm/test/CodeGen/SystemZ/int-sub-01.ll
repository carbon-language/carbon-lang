; Test 32-bit subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check SR.
define i32 @f1(i32 %a, i32 %b) {
; CHECK: f1:
; CHECK: sr %r2, %r3
; CHECK: br %r14
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the low end of the S range.
define i32 @f2(i32 %a, i32 *%src) {
; CHECK: f2:
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32 *%src
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the aligned S range.
define i32 @f3(i32 %a, i32 *%src) {
; CHECK: f3:
; CHECK: s %r2, 4092(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 1023
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word up, which should use SY instead of S.
define i32 @f4(i32 %a, i32 *%src) {
; CHECK: f4:
; CHECK: sy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 1024
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the aligned SY range.
define i32 @f5(i32 %a, i32 *%src) {
; CHECK: f5:
; CHECK: sy %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 %a, i32 *%src) {
; CHECK: f6:
; CHECK: agfi %r3, 524288
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the high end of the negative aligned SY range.
define i32 @f7(i32 %a, i32 *%src) {
; CHECK: f7:
; CHECK: sy %r2, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the low end of the SY range.
define i32 @f8(i32 %a, i32 *%src) {
; CHECK: f8:
; CHECK: sy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i32 %a, i32 *%src) {
; CHECK: f9:
; CHECK: agfi %r3, -524292
; CHECK: s %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check that S allows an index.
define i32 @f10(i32 %a, i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: s %r2, 4092({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check that SY allows an index.
define i32 @f11(i32 %a, i64 %src, i64 %index) {
; CHECK: f11:
; CHECK: sy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32 *%ptr
  %sub = sub i32 %a, %b
  ret i32 %sub
}
