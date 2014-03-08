; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=generic | FileCheck %s

; Check that we recognize this idiom for rotation too:
;    a << (b & (OpSize-1)) | a >> ((0 - b) & (OpSize-1))

define i32 @rotate_left_32(i32 %a, i32 %b) {
; CHECK-LABEL: rotate_left_32:
; CHECK: roll
entry:
  %and = and i32 %b, 31
  %shl = shl i32 %a, %and
  %0 = sub i32 0, %b
  %and3 = and i32 %0, 31
  %shr = lshr i32 %a, %and3
  %or = or i32 %shl, %shr
  ret i32 %or
}

define i32 @rotate_right_32(i32 %a, i32 %b) {
; CHECK-LABEL: rotate_right_32:
; CHECK: rorl
entry:
  %and = and i32 %b, 31
  %shl = lshr i32 %a, %and
  %0 = sub i32 0, %b
  %and3 = and i32 %0, 31
  %shr = shl i32 %a, %and3
  %or = or i32 %shl, %shr
  ret i32 %or
}

define i64 @rotate_left_64(i64 %a, i64 %b) {
; CHECK-LABEL: rotate_left_64:
; CHECK: rolq
entry:
  %and = and i64 %b, 63
  %shl = shl i64 %a, %and
  %0 = sub i64 0, %b
  %and3 = and i64 %0, 63
  %shr = lshr i64 %a, %and3
  %or = or i64 %shl, %shr
  ret i64 %or
}

define i64 @rotate_right_64(i64 %a, i64 %b) {
; CHECK-LABEL: rotate_right_64:
; CHECK: rorq
entry:
  %and = and i64 %b, 63
  %shl = lshr i64 %a, %and
  %0 = sub i64 0, %b
  %and3 = and i64 %0, 63
  %shr = shl i64 %a, %and3
  %or = or i64 %shl, %shr
  ret i64 %or
}

; Also check mem operand.

define void @rotate_left_m32(i32 *%pa, i32 %b) {
; CHECK-LABEL: rotate_left_m32:
; CHECK: roll
; no store:
; CHECK-NOT: mov
entry:
  %a = load i32* %pa, align 16
  %and = and i32 %b, 31
  %shl = shl i32 %a, %and
  %0 = sub i32 0, %b
  %and3 = and i32 %0, 31
  %shr = lshr i32 %a, %and3
  %or = or i32 %shl, %shr
  store i32 %or, i32* %pa, align 32
  ret void
}

define void @rotate_right_m32(i32 *%pa, i32 %b) {
; CHECK-LABEL: rotate_right_m32:
; CHECK: rorl
; no store:
; CHECK-NOT: mov
entry:
  %a = load i32* %pa, align 16
  %and = and i32 %b, 31
  %shl = lshr i32 %a, %and
  %0 = sub i32 0, %b
  %and3 = and i32 %0, 31
  %shr = shl i32 %a, %and3
  %or = or i32 %shl, %shr
  store i32 %or, i32* %pa, align 32
  ret void
}

define void @rotate_left_m64(i64 *%pa, i64 %b) {
; CHECK-LABEL: rotate_left_m64:
; CHECK: rolq
; no store:
; CHECK-NOT: mov
entry:
  %a = load i64* %pa, align 16
  %and = and i64 %b, 63
  %shl = shl i64 %a, %and
  %0 = sub i64 0, %b
  %and3 = and i64 %0, 63
  %shr = lshr i64 %a, %and3
  %or = or i64 %shl, %shr
  store i64 %or, i64* %pa, align 64
  ret void
}

define void @rotate_right_m64(i64 *%pa, i64 %b) {
; CHECK-LABEL: rotate_right_m64:
; CHECK: rorq
; no store:
; CHECK-NOT: mov
entry:
  %a = load i64* %pa, align 16
  %and = and i64 %b, 63
  %shl = lshr i64 %a, %and
  %0 = sub i64 0, %b
  %and3 = and i64 %0, 63
  %shr = shl i64 %a, %and3
  %or = or i64 %shl, %shr
  store i64 %or, i64* %pa, align 64
  ret void
}
