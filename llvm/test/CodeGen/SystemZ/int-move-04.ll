; Test 8-bit GPR stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test an i8 store, which should get converted into an i32 truncation.
define void @f1(i8 *%dst, i8 %val) {
; CHECK-LABEL: f1:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  store i8 %val, i8 *%dst
  ret void
}

; Test an i32 truncating store.
define void @f2(i8 *%dst, i32 %val) {
; CHECK-LABEL: f2:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %trunc = trunc i32 %val to i8
  store i8 %trunc, i8 *%dst
  ret void
}

; Test an i64 truncating store.
define void @f3(i8 *%dst, i64 %val) {
; CHECK-LABEL: f3:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %trunc = trunc i64 %val to i8
  store i8 %trunc, i8 *%dst
  ret void
}

; Check the high end of the STC range.
define void @f4(i8 *%dst, i8 %val) {
; CHECK-LABEL: f4:
; CHECK: stc %r3, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 4095
  store i8 %val, i8 *%ptr
  ret void
}

; Check the next byte up, which should use STCY instead of STC.
define void @f5(i8 *%dst, i8 %val) {
; CHECK-LABEL: f5:
; CHECK: stcy %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 4096
  store i8 %val, i8 *%ptr
  ret void
}

; Check the high end of the STCY range.
define void @f6(i8 *%dst, i8 %val) {
; CHECK-LABEL: f6:
; CHECK: stcy %r3, 524287(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 524287
  store i8 %val, i8 *%ptr
  ret void
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f7(i8 *%dst, i8 %val) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 524288
  store i8 %val, i8 *%ptr
  ret void
}

; Check the high end of the negative STCY range.
define void @f8(i8 *%dst, i8 %val) {
; CHECK-LABEL: f8:
; CHECK: stcy %r3, -1(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 -1
  store i8 %val, i8 *%ptr
  ret void
}

; Check the low end of the STCY range.
define void @f9(i8 *%dst, i8 %val) {
; CHECK-LABEL: f9:
; CHECK: stcy %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 -524288
  store i8 %val, i8 *%ptr
  ret void
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f10(i8 *%dst, i8 %val) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524289
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8 *%dst, i64 -524289
  store i8 %val, i8 *%ptr
  ret void
}

; Check that STC allows an index.
define void @f11(i64 %dst, i64 %index, i8 %val) {
; CHECK-LABEL: f11:
; CHECK: stc %r4, 4095(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to i8 *
  store i8 %val, i8 *%ptr
  ret void
}

; Check that STCY allows an index.
define void @f12(i64 %dst, i64 %index, i8 %val) {
; CHECK-LABEL: f12:
; CHECK: stcy %r4, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i8 *
  store i8 %val, i8 *%ptr
  ret void
}
