; Test sequences that can use RISBG with a normal first operand.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test a case with two ANDs.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: risbg %r2, %r3, 60, 62, 0
; CHECK: br %r14
  %anda = and i32 %a, -15
  %andb = and i32 %b, 14
  %or = or i32 %anda, %andb
  ret i32 %or
}

; ...and again with i64.
define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: risbg %r2, %r3, 60, 62, 0
; CHECK: br %r14
  %anda = and i64 %a, -15
  %andb = and i64 %b, 14
  %or = or i64 %anda, %andb
  ret i64 %or
}

; Test a case with two ANDs and a shift.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: risbg %r2, %r3, 60, 63, 56
; CHECK: br %r14
  %anda = and i32 %a, -16
  %shr = lshr i32 %b, 8
  %andb = and i32 %shr, 15
  %or = or i32 %anda, %andb
  ret i32 %or
}

; ...and again with i64.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: risbg %r2, %r3, 60, 63, 56
; CHECK: br %r14
  %anda = and i64 %a, -16
  %shr = lshr i64 %b, 8
  %andb = and i64 %shr, 15
  %or = or i64 %anda, %andb
  ret i64 %or
}

; Test a case with a single AND and a left shift.
define i32 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: risbg %r2, %r3, 32, 53, 10
; CHECK: br %r14
  %anda = and i32 %a, 1023
  %shlb = shl i32 %b, 10
  %or = or i32 %anda, %shlb
  ret i32 %or
}

; ...and again with i64.
define i64 @f6(i64 %a, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: risbg %r2, %r3, 0, 53, 10
; CHECK: br %r14
  %anda = and i64 %a, 1023
  %shlb = shl i64 %b, 10
  %or = or i64 %anda, %shlb
  ret i64 %or
}

; Test a case with a single AND and a right shift.
define i32 @f7(i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: risbg %r2, %r3, 40, 63, 56
; CHECK: br %r14
  %anda = and i32 %a, -16777216
  %shrb = lshr i32 %b, 8
  %or = or i32 %anda, %shrb
  ret i32 %or
}

; ...and again with i64.
define i64 @f8(i64 %a, i64 %b) {
; CHECK-LABEL: f8:
; CHECK: risbg %r2, %r3, 8, 63, 56
; CHECK: br %r14
  %anda = and i64 %a, -72057594037927936
  %shrb = lshr i64 %b, 8
  %or = or i64 %anda, %shrb
  ret i64 %or
}
