; Test sign extensions from a halfword to an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: lhr %r2, %r2
; CHECk: br %r14
  %half = trunc i32 %a to i16
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; ...and again with an i64.
define i32 @f2(i64 %a) {
; CHECK: f2:
; CHECK: lhr %r2, %r2
; CHECk: br %r14
  %half = trunc i64 %a to i16
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LH range.
define i32 @f3(i16 *%src) {
; CHECK: f3:
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16 *%src
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LH range.
define i32 @f4(i16 *%src) {
; CHECK: f4:
; CHECK: lh %r2, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 2047
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs LHY rather than LH.
define i32 @f5(i16 *%src) {
; CHECK: f5:
; CHECK: lhy %r2, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 2048
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LHY range.
define i32 @f6(i16 *%src) {
; CHECK: f6:
; CHECK: lhy %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262143
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f7(i16 *%src) {
; CHECK: f7:
; CHECK: agfi %r2, 524288
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 262144
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the negative LHY range.
define i32 @f8(i16 *%src) {
; CHECK: f8:
; CHECK: lhy %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -1
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LHY range.
define i32 @f9(i16 *%src) {
; CHECK: f9:
; CHECK: lhy %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262144
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f10(i16 *%src) {
; CHECK: f10:
; CHECK: agfi %r2, -524290
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16 *%src, i64 -262145
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check that LH allows an index
define i32 @f11(i64 %src, i64 %index) {
; CHECK: f11:
; CHECK: lh %r2, 4094(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check that LH allows an index
define i32 @f12(i64 %src, i64 %index) {
; CHECK: f12:
; CHECK: lhy %r2, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  %half = load i16 *%ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}
