; RUN: llc -mtriple=arm64-apple-ios < %s | FileCheck %s

define i64 @sbfiz64(i64 %v) {
; CHECK-LABEL: sbfiz64:
; CHECK: sbfiz	x0, x0, #1, #16
  %shl = shl i64 %v, 48
  %shr = ashr i64 %shl, 47
  ret i64 %shr
}

define i32 @sbfiz32(i32 %v) {
; CHECK-LABEL: sbfiz32:
; CHECK: sbfiz	w0, w0, #1, #14
  %shl = shl i32 %v, 18
  %shr = ashr i32 %shl, 17
  ret i32 %shr
}

define i64 @ubfiz64(i64 %v) {
; CHECK-LABEL: ubfiz64:
; CHECK: ubfiz	x0, x0, #36, #11
  %shl = shl i64 %v, 53
  %shr = lshr i64 %shl, 17
  ret i64 %shr
}

define i32 @ubfiz32(i32 %v) {
; CHECK-LABEL: ubfiz32:
; CHECK: ubfiz	w0, w0, #6, #24
  %shl = shl i32 %v, 8
  %shr = lshr i32 %shl, 2
  ret i32 %shr
}
