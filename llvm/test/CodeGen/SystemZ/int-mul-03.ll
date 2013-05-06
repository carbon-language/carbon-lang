; Test multiplications between an i64 and a sign-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check MSGFR.
define i64 @f1(i64 %a, i32 %b) {
; CHECK: f1:
; CHECK: msgfr %r2, %r3
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check MSGF with no displacement.
define i64 @f2(i64 %a, i32 *%src) {
; CHECK: f2:
; CHECK: msgf %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32 *%src
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the high end of the aligned MSGF range.
define i64 @f3(i64 %a, i32 *%src) {
; CHECK: f3:
; CHECK: msgf %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i32 *%src) {
; CHECK: f4:
; CHECK: agfi %r3, 524288
; CHECK: msgf %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the high end of the negative aligned MSGF range.
define i64 @f5(i64 %a, i32 *%src) {
; CHECK: f5:
; CHECK: msgf %r2, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the low end of the MSGF range.
define i64 @f6(i64 %a, i32 *%src) {
; CHECK: f6:
; CHECK: msgf %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, i32 *%src) {
; CHECK: f7:
; CHECK: agfi %r3, -524292
; CHECK: msgf %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

; Check that MSGF allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK: f8:
; CHECK: msgf %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32 *%ptr
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}
