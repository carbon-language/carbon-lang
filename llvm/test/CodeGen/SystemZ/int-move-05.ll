; Test 16-bit GPR stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test an i16 store, which should get converted into an i32 truncation.
define void @f1(i16 *%dst, i16 %val) {
; CHECK-LABEL: f1:
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  store i16 %val, i16 *%dst
  ret void
}

; Test an i32 truncating store.
define void @f2(i16 *%dst, i32 %val) {
; CHECK-LABEL: f2:
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %trunc = trunc i32 %val to i16
  store i16 %trunc, i16 *%dst
  ret void
}

; Test an i64 truncating store.
define void @f3(i16 *%dst, i64 %val) {
; CHECK-LABEL: f3:
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %trunc = trunc i64 %val to i16
  store i16 %trunc, i16 *%dst
  ret void
}

; Check the high end of the STH range.
define void @f4(i16 *%dst, i16 %val) {
; CHECK-LABEL: f4:
; CHECK: sth %r3, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 2047
  store i16 %val, i16 *%ptr
  ret void
}

; Check the next halfword up, which should use STHY instead of STH.
define void @f5(i16 *%dst, i16 %val) {
; CHECK-LABEL: f5:
; CHECK: sthy %r3, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 2048
  store i16 %val, i16 *%ptr
  ret void
}

; Check the high end of the aligned STHY range.
define void @f6(i16 *%dst, i16 %val) {
; CHECK-LABEL: f6:
; CHECK: sthy %r3, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 262143
  store i16 %val, i16 *%ptr
  ret void
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f7(i16 *%dst, i16 %val) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 262144
  store i16 %val, i16 *%ptr
  ret void
}

; Check the high end of the negative aligned STHY range.
define void @f8(i16 *%dst, i16 %val) {
; CHECK-LABEL: f8:
; CHECK: sthy %r3, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 -1
  store i16 %val, i16 *%ptr
  ret void
}

; Check the low end of the STHY range.
define void @f9(i16 *%dst, i16 %val) {
; CHECK-LABEL: f9:
; CHECK: sthy %r3, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 -262144
  store i16 %val, i16 *%ptr
  ret void
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f10(i16 *%dst, i16 %val) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524290
; CHECK: sth %r3, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, i16 *%dst, i64 -262145
  store i16 %val, i16 *%ptr
  ret void
}

; Check that STH allows an index.
define void @f11(i64 %dst, i64 %index, i16 %val) {
; CHECK-LABEL: f11:
; CHECK: sth %r4, 4094({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to i16 *
  store i16 %val, i16 *%ptr
  ret void
}

; Check that STHY allows an index.
define void @f12(i64 %dst, i64 %index, i16 %val) {
; CHECK-LABEL: f12:
; CHECK: sthy %r4, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %dst, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i16 *
  store i16 %val, i16 *%ptr
  ret void
}
