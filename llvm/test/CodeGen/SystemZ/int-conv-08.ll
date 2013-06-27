; Test zero extensions from a halfword to an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i64 @f1(i32 %a) {
; CHECK: f1:
; CHECK: llghr %r2, %r2
; CHECK: br %r14
  %half = trunc i32 %a to i16
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; ...and again with an i64.
define i64 @f2(i64 %a) {
; CHECK: f2:
; CHECK: llghr %r2, %r2
; CHECK: br %r14
  %half = trunc i64 %a to i16
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check ANDs that are equivalent to zero extension.
define i64 @f3(i64 %a) {
; CHECK: f3:
; CHECK: llghr %r2, %r2
; CHECK: br %r14
  %ext = and i64 %a, 65535
  ret i64 %ext
}

; Check LLGH with no displacement.
define i64 @f4(i16 *%src) {
; CHECK: f4:
; CHECK: llgh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16 *%src
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check the high end of the LLGH range.
define i64 @f5(i16 *%src) {
; CHECK: f5:
; CHECK: llgh %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262143
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i16 *%src) {
; CHECK: f6:
; CHECK: agfi %r2, 524288
; CHECK: llgh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262144
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check the high end of the negative LLGH range.
define i64 @f7(i16 *%src) {
; CHECK: f7:
; CHECK: llgh %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -1
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check the low end of the LLGH range.
define i64 @f8(i16 *%src) {
; CHECK: f8:
; CHECK: llgh %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262144
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f9(i16 *%src) {
; CHECK: f9:
; CHECK: agfi %r2, -524290
; CHECK: llgh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262145
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}

; Check that LLGH allows an index
define i64 @f10(i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: llgh %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16 *%ptr
  %ext = zext i16 %half to i64
  ret i64 %ext
}
