; RUN: llc -fast-isel -fast-isel-abort -aarch64-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s

define i32 @icmp_eq_i1(i1 signext %a) {
; CHECK-LABEL: icmp_eq_i1
; CHECK:       cbz w0, {{LBB.+_2}}
  %1 = icmp eq i1 %a, 0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i8(i8 signext %a) {
; CHECK-LABEL: icmp_eq_i8
; CHECK:       cbz w0, {{LBB.+_2}}
  %1 = icmp eq i8 %a, 0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i16(i16 signext %a) {
; CHECK-LABEL: icmp_eq_i16
; CHECK:       cbz w0, {{LBB.+_2}}
  %1 = icmp eq i16 %a, 0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i32(i32 %a) {
; CHECK-LABEL: icmp_eq_i32
; CHECK:       cbz w0, {{LBB.+_2}}
  %1 = icmp eq i32 %a, 0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

define i32 @icmp_eq_i64(i64 %a) {
; CHECK-LABEL: icmp_eq_i64
; CHECK:       cbz x0, {{LBB.+_2}}
  %1 = icmp eq i64 %a, 0
  br i1 %1, label %bb1, label %bb2
bb2:
  ret i32 1
bb1:
  ret i32 0
}

