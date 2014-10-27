; RUN: llc                             -aarch64-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -fast-isel -fast-isel-abort -aarch64-atomic-cfg-tidy=0 -verify-machineinstrs -mtriple=aarch64-apple-darwin < %s | FileCheck %s

define i32 @icmp_eq_i8(i8 zeroext %a) {
; CHECK-LABEL: icmp_eq_i8
; CHECK:       tbz {{w[0-9]+}}, #0, {{LBB.+_2}}
  %1 = and i8 %a, 1
  %2 = icmp eq i8 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i16(i16 zeroext %a) {
; CHECK-LABEL: icmp_eq_i16
; CHECK:       tbz w0, #1, {{LBB.+_2}}
  %1 = and i16 %a, 2
  %2 = icmp eq i16 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i32(i32 %a) {
; CHECK-LABEL: icmp_eq_i32
; CHECK:       tbz w0, #2, {{LBB.+_2}}
  %1 = and i32 %a, 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i64_1(i64 %a) {
; CHECK-LABEL: icmp_eq_i64_1
; CHECK:       tbz w0, #3, {{LBB.+_2}}
  %1 = and i64 %a, 8
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_eq_i64_2(i64 %a) {
; CHECK-LABEL: icmp_eq_i64_2
; CHECK:       tbz x0, #32, {{LBB.+_2}}
  %1 = and i64 %a, 4294967296
  %2 = icmp eq i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i8(i8 zeroext %a) {
; CHECK-LABEL: icmp_ne_i8
; CHECK:       tbnz w0, #0, {{LBB.+_2}}
  %1 = and i8 %a, 1
  %2 = icmp ne i8 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i16(i16 zeroext %a) {
; CHECK-LABEL: icmp_ne_i16
; CHECK:       tbnz w0, #1, {{LBB.+_2}}
  %1 = and i16 %a, 2
  %2 = icmp ne i16 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i32(i32 %a) {
; CHECK-LABEL: icmp_ne_i32
; CHECK:       tbnz w0, #2, {{LBB.+_2}}
  %1 = and i32 %a, 4
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i64_1(i64 %a) {
; CHECK-LABEL: icmp_ne_i64_1
; CHECK:       tbnz w0, #3, {{LBB.+_2}}
  %1 = and i64 %a, 8
  %2 = icmp ne i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

define i32 @icmp_ne_i64_2(i64 %a) {
; CHECK-LABEL: icmp_ne_i64_2
; CHECK:       tbnz x0, #32, {{LBB.+_2}}
  %1 = and i64 %a, 4294967296
  %2 = icmp ne i64 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

; Test that we don't fold the 'and' instruction into the compare.
define i32 @icmp_eq_and_i32(i32 %a, i1 %c) {
; CHECK-LABEL: icmp_eq_and_i32
; CHECK:       and  [[REG:w[0-9]+]], w0, #0x4
; CHECK-NEXT:  cbz  [[REG]], {{LBB.+_3}}
  %1 = and i32 %a, 4
  br i1 %c, label %bb0, label %bb2
bb0:
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %bb1, label %bb2, !prof !0
bb1:
  ret i32 1
bb2:
  ret i32 0
}

!0 = metadata !{metadata !"branch_weights", i32 0, i32 2147483647}
!1 = metadata !{metadata !"branch_weights", i32 2147483647, i32 0}
