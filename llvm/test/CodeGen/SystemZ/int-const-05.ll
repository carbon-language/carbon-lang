; Test moves of integers to 4-byte memory locations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check moves of zero.
define void @f1(i32 *%a) {
; CHECK: f1:
; CHECK: mvhi 0(%r2), 0
; CHECK: br %r14
  store i32 0, i32 *%a
  ret void
}

; Check the high end of the signed 16-bit range.
define void @f2(i32 *%a) {
; CHECK: f2:
; CHECK: mvhi 0(%r2), 32767
; CHECK: br %r14
  store i32 32767, i32 *%a
  ret void
}

; Check the next value up, which can't use MVHI.
define void @f3(i32 *%a) {
; CHECK: f3:
; CHECK-NOT: mvhi
; CHECK: br %r14
  store i32 32768, i32 *%a
  ret void
}

; Check moves of -1.
define void @f4(i32 *%a) {
; CHECK: f4:
; CHECK: mvhi 0(%r2), -1
; CHECK: br %r14
  store i32 -1, i32 *%a
  ret void
}

; Check the low end of the MVHI range.
define void @f5(i32 *%a) {
; CHECK: f5:
; CHECK: mvhi 0(%r2), -32768
; CHECK: br %r14
  store i32 -32768, i32 *%a
  ret void
}

; Check the next value down, which can't use MVHI.
define void @f6(i32 *%a) {
; CHECK: f6:
; CHECK-NOT: mvhi
; CHECK: br %r14
  store i32 -32769, i32 *%a
  ret void
}

; Check the high end of the MVHI range.
define void @f7(i32 *%a) {
; CHECK: f7:
; CHECK: mvhi 4092(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i32 *%a, i64 1023
  store i32 42, i32 *%ptr
  ret void
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(i32 *%a) {
; CHECK: f8:
; CHECK: aghi %r2, 4096
; CHECK: mvhi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i32 *%a, i64 1024
  store i32 42, i32 *%ptr
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f9(i32 *%a) {
; CHECK: f9:
; CHECK: aghi %r2, -4
; CHECK: mvhi 0(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i32 *%a, i64 -1
  store i32 42, i32 *%ptr
  ret void
}

; Check that MVHI does not allow an index
define void @f10(i64 %src, i64 %index) {
; CHECK: f10:
; CHECK: agr %r2, %r3
; CHECK: mvhi 0(%r2), 42
; CHECK: br %r14
  %add = add i64 %src, %index
  %ptr = inttoptr i64 %add to i32 *
  store i32 42, i32 *%ptr
  ret void
}
