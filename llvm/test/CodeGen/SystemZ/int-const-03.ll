; Test moves of integers to byte memory locations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the unsigned range.
define void @f1(i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: mvi 0(%r2), 0
; CHECK: br %r14
  store i8 0, i8 *%ptr
  ret void
}

; Check the high end of the signed range.
define void @f2(i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: mvi 0(%r2), 127
; CHECK: br %r14
  store i8 127, i8 *%ptr
  ret void
}

; Check the next value up.
define void @f3(i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: mvi 0(%r2), 128
; CHECK: br %r14
  store i8 -128, i8 *%ptr
  ret void
}

; Check the high end of the unsigned range.
define void @f4(i8 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: mvi 0(%r2), 255
; CHECK: br %r14
  store i8 255, i8 *%ptr
  ret void
}

; Check -1.
define void @f5(i8 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: mvi 0(%r2), 255
; CHECK: br %r14
  store i8 -1, i8 *%ptr
  ret void
}

; Check the low end of the signed range.
define void @f6(i8 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: mvi 0(%r2), 128
; CHECK: br %r14
  store i8 -128, i8 *%ptr
  ret void
}

; Check the next value down.
define void @f7(i8 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: mvi 0(%r2), 127
; CHECK: br %r14
  store i8 -129, i8 *%ptr
  ret void
}

; Check the high end of the MVI range.
define void @f8(i8 *%src) {
; CHECK-LABEL: f8:
; CHECK: mvi 4095(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 4095
  store i8 42, i8 *%ptr
  ret void
}

; Check the next byte up, which should use MVIY instead of MVI.
define void @f9(i8 *%src) {
; CHECK-LABEL: f9:
; CHECK: mviy 4096(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 4096
  store i8 42, i8 *%ptr
  ret void
}

; Check the high end of the MVIY range.
define void @f10(i8 *%src) {
; CHECK-LABEL: f10:
; CHECK: mviy 524287(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 524287
  store i8 42, i8 *%ptr
  ret void
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11(i8 *%src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r2, 524288
; CHECK: mvi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 524288
  store i8 42, i8 *%ptr
  ret void
}

; Check the high end of the negative MVIY range.
define void @f12(i8 *%src) {
; CHECK-LABEL: f12:
; CHECK: mviy -1(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -1
  store i8 42, i8 *%ptr
  ret void
}

; Check the low end of the MVIY range.
define void @f13(i8 *%src) {
; CHECK-LABEL: f13:
; CHECK: mviy -524288(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -524288
  store i8 42, i8 *%ptr
  ret void
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f14(i8 *%src) {
; CHECK-LABEL: f14:
; CHECK: agfi %r2, -524289
; CHECK: mvi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i8 *%src, i64 -524289
  store i8 42, i8 *%ptr
  ret void
}

; Check that MVI does not allow an index.  We prefer STC in that case.
define void @f15(i64 %src, i64 %index) {
; CHECK-LABEL: f15:
; CHECK: lhi [[TMP:%r[0-5]]], 42
; CHECK: stc [[TMP]], 4095({{%r2,%r3|%r3,%r2}}
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to i8 *
  store i8 42, i8 *%ptr
  ret void
}

; Check that MVIY does not allow an index.  We prefer STCY in that case.
define void @f16(i64 %src, i64 %index) {
; CHECK-LABEL: f16:
; CHECK: lhi [[TMP:%r[0-5]]], 42
; CHECK: stcy [[TMP]], 4096({{%r2,%r3|%r3,%r2}}
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i8 *
  store i8 42, i8 *%ptr
  ret void
}
