; Test zero extensions from a halfword to an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %half = trunc i32 %a to i16
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; ...and again with an i64.
define i32 @f2(i64 %a) {
; CHECK: f2:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %half = trunc i64 %a to i16
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check ANDs that are equivalent to zero extension.
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: llhr %r2, %r2
; CHECK: br %r14
  %ext = and i32 %a, 65535
  ret i32 %ext
}

; Check LLH with no displacement.
define i32 @f4(i16 *%src) {
; CHECK: f4:
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16 *%src
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LLH range.
define i32 @f5(i16 *%src) {
; CHECK: f5:
; CHECK: llh %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262143
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i16 *%src) {
; CHECK: f6:
; CHECK: agfi %r2, 524288
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262144
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the negative LLH range.
define i32 @f7(i16 *%src) {
; CHECK: f7:
; CHECK: llh %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -1
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LLH range.
define i32 @f8(i16 *%src) {
; CHECK: f8:
; CHECK: llh %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262144
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i16 *%src) {
; CHECK: f9:
; CHECK: agfi %r2, -524290
; CHECK: llh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262145
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}

; Check that LLH allows an index
define i32 @f10(i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: llh %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16 *%ptr
  %ext = zext i16 %half to i32
  ret i32 %ext
}
