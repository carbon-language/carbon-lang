; Test zero extensions from a byte to an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: llcr %r2, %r2
; CHECK: br %r14
  %byte = trunc i32 %a to i8
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; ...and again with an i64.
define i32 @f2(i64 %a) {
; CHECK: f2:
; CHECK: llcr %r2, %r2
; CHECK: br %r14
  %byte = trunc i64 %a to i8
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check ANDs that are equivalent to zero extension.
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: llcr %r2, %r2
; CHECK: br %r14
  %ext = and i32 %a, 255
  ret i32 %ext
}

; Check LLC with no displacement.
define i32 @f4(i8 *%src) {
; CHECK: f4:
; CHECK: llc %r2, 0(%r2)
; CHECK: br %r14
  %byte = load i8 *%src
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check the high end of the LLC range.
define i32 @f5(i8 *%src) {
; CHECK: f5:
; CHECK: llc %r2, 524287(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 524287
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i8 *%src) {
; CHECK: f6:
; CHECK: agfi %r2, 524288
; CHECK: llc %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 524288
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check the high end of the negative LLC range.
define i32 @f7(i8 *%src) {
; CHECK: f7:
; CHECK: llc %r2, -1(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -1
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check the low end of the LLC range.
define i32 @f8(i8 *%src) {
; CHECK: f8:
; CHECK: llc %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -524288
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i8 *%src) {
; CHECK: f9:
; CHECK: agfi %r2, -524289
; CHECK: llc %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -524289
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}

; Check that LLC allows an index
define i32 @f10(i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: llc %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i8 *
  %byte = load i8 *%ptr
  %ext = zext i8 %byte to i32
  ret i32 %ext
}
