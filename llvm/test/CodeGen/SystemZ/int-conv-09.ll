; Test sign extensions from an i32 to an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i64 @f1(i32 %a) {
; CHECK: f1:
; CHECK: lgfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %a to i64
  ret i64 %ext
}

; ...and again with an i64.
define i64 @f2(i64 %a) {
; CHECK: f2:
; CHECK: lgfr %r2, %r2
; CHECK: br %r14
  %word = trunc i64 %a to i32
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check LGF with no displacement.
define i64 @f3(i32 *%src) {
; CHECK: f3:
; CHECK: lgf %r2, 0(%r2)
; CHECK: br %r14
  %word = load i32 *%src
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check the high end of the LGF range.
define i64 @f4(i32 *%src) {
; CHECK: f4:
; CHECK: lgf %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131071
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f5(i32 *%src) {
; CHECK: f5:
; CHECK: agfi %r2, 524288
; CHECK: lgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 131072
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check the high end of the negative LGF range.
define i64 @f6(i32 *%src) {
; CHECK: f6:
; CHECK: lgf %r2, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -1
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check the low end of the LGF range.
define i64 @f7(i32 *%src) {
; CHECK: f7:
; CHECK: lgf %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131072
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(i32 *%src) {
; CHECK: f8:
; CHECK: agfi %r2, -524292
; CHECK: lgf %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32 *%src, i64 -131073
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Check that LGF allows an index.
define i64 @f9(i64 %src, i64 %index) {
; CHECK: f9:
; CHECK: lgf %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %word = load i32 *%ptr
  %ext = sext i32 %word to i64
  ret i64 %ext
}

; Test a case where we spill the source of at least one LGFR.  We want
; to use LGF if possible.
define void @f10(i64 *%ptr1, i32 *%ptr2) {
; CHECK: f10:
; CHECK: lgf {{%r[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile i32 *%ptr2
  %val1 = load volatile i32 *%ptr2
  %val2 = load volatile i32 *%ptr2
  %val3 = load volatile i32 *%ptr2
  %val4 = load volatile i32 *%ptr2
  %val5 = load volatile i32 *%ptr2
  %val6 = load volatile i32 *%ptr2
  %val7 = load volatile i32 *%ptr2
  %val8 = load volatile i32 *%ptr2
  %val9 = load volatile i32 *%ptr2
  %val10 = load volatile i32 *%ptr2
  %val11 = load volatile i32 *%ptr2
  %val12 = load volatile i32 *%ptr2
  %val13 = load volatile i32 *%ptr2
  %val14 = load volatile i32 *%ptr2
  %val15 = load volatile i32 *%ptr2

  %ext0 = sext i32 %val0 to i64
  %ext1 = sext i32 %val1 to i64
  %ext2 = sext i32 %val2 to i64
  %ext3 = sext i32 %val3 to i64
  %ext4 = sext i32 %val4 to i64
  %ext5 = sext i32 %val5 to i64
  %ext6 = sext i32 %val6 to i64
  %ext7 = sext i32 %val7 to i64
  %ext8 = sext i32 %val8 to i64
  %ext9 = sext i32 %val9 to i64
  %ext10 = sext i32 %val10 to i64
  %ext11 = sext i32 %val11 to i64
  %ext12 = sext i32 %val12 to i64
  %ext13 = sext i32 %val13 to i64
  %ext14 = sext i32 %val14 to i64
  %ext15 = sext i32 %val15 to i64

  store volatile i32 %val0, i32 *%ptr2
  store volatile i32 %val1, i32 *%ptr2
  store volatile i32 %val2, i32 *%ptr2
  store volatile i32 %val3, i32 *%ptr2
  store volatile i32 %val4, i32 *%ptr2
  store volatile i32 %val5, i32 *%ptr2
  store volatile i32 %val6, i32 *%ptr2
  store volatile i32 %val7, i32 *%ptr2
  store volatile i32 %val8, i32 *%ptr2
  store volatile i32 %val9, i32 *%ptr2
  store volatile i32 %val10, i32 *%ptr2
  store volatile i32 %val11, i32 *%ptr2
  store volatile i32 %val12, i32 *%ptr2
  store volatile i32 %val13, i32 *%ptr2
  store volatile i32 %val14, i32 *%ptr2
  store volatile i32 %val15, i32 *%ptr2

  store volatile i64 %ext0, i64 *%ptr1
  store volatile i64 %ext1, i64 *%ptr1
  store volatile i64 %ext2, i64 *%ptr1
  store volatile i64 %ext3, i64 *%ptr1
  store volatile i64 %ext4, i64 *%ptr1
  store volatile i64 %ext5, i64 *%ptr1
  store volatile i64 %ext6, i64 *%ptr1
  store volatile i64 %ext7, i64 *%ptr1
  store volatile i64 %ext8, i64 *%ptr1
  store volatile i64 %ext9, i64 *%ptr1
  store volatile i64 %ext10, i64 *%ptr1
  store volatile i64 %ext11, i64 *%ptr1
  store volatile i64 %ext12, i64 *%ptr1
  store volatile i64 %ext13, i64 *%ptr1
  store volatile i64 %ext14, i64 *%ptr1
  store volatile i64 %ext15, i64 *%ptr1

  ret void
}
