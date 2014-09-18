; RUN: llc -fast-isel -fast-isel-abort -aarch64-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s

define i32 @icmp_eq_i8(i8 zeroext %a) {
; CHECK-LABEL: icmp_eq_i8
; CHECK:       tbz w0, #0, {{LBB.+_2}}
  %1 = and i8 %a, 1
  %2 = icmp eq i8 %1, 0
  br i1 %2, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i16(i16 zeroext %a) {
; CHECK-LABEL: icmp_eq_i16
; CHECK:       tbz w0, #1, {{LBB.+_2}}
  %1 = and i16 %a, 2
  %2 = icmp eq i16 %1, 0
  br i1 %2, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i32(i32 %a) {
; CHECK-LABEL: icmp_eq_i32
; CHECK:       tbz w0, #2, {{LBB.+_2}}
  %1 = and i32 %a, 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i64(i64 %a) {
; CHECK-LABEL: icmp_eq_i64
; CHECK:       tbz x0, #3, {{LBB.+_2}}
  %1 = and i64 %a, 8
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

