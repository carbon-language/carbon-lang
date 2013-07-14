; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Note that this should be refactored (for efficiency if nothing else)
; when the PCS is implemented so we don't have to worry about the
; loads and stores.

@var_i32 = global i32 42
@var_i64 = global i64 0

; Add pure 12-bit immediates:
define void @add_small() {
; CHECK-LABEL: add_small:

; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #4095
  %val32 = load i32* @var_i32
  %newval32 = add i32 %val32, 4095
  store i32 %newval32, i32* @var_i32

; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #52
  %val64 = load i64* @var_i64
  %newval64 = add i64 %val64, 52
  store i64 %newval64, i64* @var_i64

  ret void
}

; Add 12-bit immediates, shifted left by 12 bits
define void @add_med() {
; CHECK-LABEL: add_med:

; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #3567, lsl #12
  %val32 = load i32* @var_i32
  %newval32 = add i32 %val32, 14610432 ; =0xdef000
  store i32 %newval32, i32* @var_i32

; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, #4095, lsl #12
  %val64 = load i64* @var_i64
  %newval64 = add i64 %val64, 16773120 ; =0xfff000
  store i64 %newval64, i64* @var_i64

  ret void
}

; Subtract 12-bit immediates
define void @sub_small() {
; CHECK-LABEL: sub_small:

; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, #4095
  %val32 = load i32* @var_i32
  %newval32 = sub i32 %val32, 4095
  store i32 %newval32, i32* @var_i32

; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, #52
  %val64 = load i64* @var_i64
  %newval64 = sub i64 %val64, 52
  store i64 %newval64, i64* @var_i64

  ret void
}

; Subtract 12-bit immediates, shifted left by 12 bits
define void @sub_med() {
; CHECK-LABEL: sub_med:

; CHECK: sub {{w[0-9]+}}, {{w[0-9]+}}, #3567, lsl #12
  %val32 = load i32* @var_i32
  %newval32 = sub i32 %val32, 14610432 ; =0xdef000
  store i32 %newval32, i32* @var_i32

; CHECK: sub {{x[0-9]+}}, {{x[0-9]+}}, #4095, lsl #12
  %val64 = load i64* @var_i64
  %newval64 = sub i64 %val64, 16773120 ; =0xfff000
  store i64 %newval64, i64* @var_i64

  ret void
}

define void @testing() {
; CHECK-LABEL: testing:
  %val = load i32* @var_i32

; CHECK: cmp {{w[0-9]+}}, #4095
; CHECK: b.ne .LBB4_6
  %cmp_pos_small = icmp ne i32 %val, 4095
  br i1 %cmp_pos_small, label %ret, label %test2

test2:
; CHECK: cmp {{w[0-9]+}}, #3567, lsl #12
; CHECK: b.lo .LBB4_6
  %newval2 = add i32 %val, 1
  store i32 %newval2, i32* @var_i32
  %cmp_pos_big = icmp ult i32 %val, 14610432
  br i1 %cmp_pos_big, label %ret, label %test3

test3:
; CHECK: cmp {{w[0-9]+}}, #123
; CHECK: b.lt .LBB4_6
  %newval3 = add i32 %val, 2
  store i32 %newval3, i32* @var_i32
  %cmp_pos_slt = icmp slt i32 %val, 123
  br i1 %cmp_pos_slt, label %ret, label %test4

test4:
; CHECK: cmp {{w[0-9]+}}, #321
; CHECK: b.gt .LBB4_6
  %newval4 = add i32 %val, 3
  store i32 %newval4, i32* @var_i32
  %cmp_pos_sgt = icmp sgt i32 %val, 321
  br i1 %cmp_pos_sgt, label %ret, label %test5

test5:
; CHECK: cmn {{w[0-9]+}}, #444
; CHECK: b.gt .LBB4_6
  %newval5 = add i32 %val, 4
  store i32 %newval5, i32* @var_i32
  %cmp_neg_uge = icmp sgt i32 %val, -444
  br i1 %cmp_neg_uge, label %ret, label %test6

test6:
  %newval6 = add i32 %val, 5
  store i32 %newval6, i32* @var_i32
  ret void

ret:
  ret void
}
; TODO: adds/subs
