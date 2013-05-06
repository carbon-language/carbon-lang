; Test moves of integers to 8-byte memory locations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check moves of zero.
define void @f1(i64 *%a) {
; CHECK: f1:
; CHECK: mvghi 0(%r2), 0
; CHECK: br %r14
  store i64 0, i64 *%a
  ret void
}

; Check the high end of the signed 16-bit range.
define void @f2(i64 *%a) {
; CHECK: f2:
; CHECK: mvghi 0(%r2), 32767
; CHECK: br %r14
  store i64 32767, i64 *%a
  ret void
}

; Check the next value up, which can't use MVGHI.
define void @f3(i64 *%a) {
; CHECK: f3:
; CHECK-NOT: mvghi
; CHECK: br %r14
  store i64 32768, i64 *%a
  ret void
}

; Check moves of -1.
define void @f4(i64 *%a) {
; CHECK: f4:
; CHECK: mvghi 0(%r2), -1
; CHECK: br %r14
  store i64 -1, i64 *%a
  ret void
}

; Check the low end of the MVGHI range.
define void @f5(i64 *%a) {
; CHECK: f5:
; CHECK: mvghi 0(%r2), -32768
; CHECK: br %r14
  store i64 -32768, i64 *%a
  ret void
}

; Check the next value down, which can't use MVGHI.
define void @f6(i64 *%a) {
; CHECK: f6:
; CHECK-NOT: mvghi
; CHECK: br %r14
  store i64 -32769, i64 *%a
  ret void
}

; Check the high end of the MVGHI range.
define void @f7(i64 *%a) {
; CHECK: f7:
; CHECK: mvghi 4088(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i64 *%a, i64 511
  store i64 42, i64 *%ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(i64 *%a) {
; CHECK: f8:
; CHECK: aghi %r2, 4096
; CHECK: mvghi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i64 *%a, i64 512
  store i64 42, i64 *%ptr
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f9(i64 *%a) {
; CHECK: f9:
; CHECK: aghi %r2, -8
; CHECK: mvghi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i64 *%a, i64 -1
  store i64 42, i64 *%ptr
  ret void
}

; Check that MVGHI does not allow an index
define void @f10(i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: agr %r2, %r3
; CHECK: mvghi 0(%r2), 42
; CHECK: br %r14
  %add = add i64 %src, %index
  %ptr = inttoptr i64 %add to i64 *
  store i64 42, i64 *%ptr
  ret void
}
