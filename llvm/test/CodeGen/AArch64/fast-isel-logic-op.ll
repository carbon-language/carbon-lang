; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel=0                  -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel=1 -fast-isel-abort -verify-machineinstrs < %s | FileCheck %s

; AND
define i32 @and_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: and_rr_i32
; CHECK:       and w0, w0, w1
  %1 = and i32 %a, %b
  ret i32 %1
}

define i64 @and_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: and_rr_i64
; CHECK:       and x0, x0, x1
  %1 = and i64 %a, %b
  ret i64 %1
}

define i32 @and_ri_i32(i32 %a) {
; CHECK-LABEL: and_ri_i32
; CHECK:       and w0, w0, #0xff
  %1 = and i32 %a, 255
  ret i32 %1
}

define i64 @and_ri_i64(i64 %a) {
; CHECK-LABEL: and_ri_i64
; CHECK:       and x0, x0, #0xff
  %1 = and i64 %a, 255
  ret i64 %1
}

define i32 @and_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: and_rs_i32
; CHECK:       and w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = and i32 %a, %1
  ret i32 %2
}

define i64 @and_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: and_rs_i64
; CHECK:       and x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = and i64 %a, %1
  ret i64 %2
}

; OR
define i32 @or_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: or_rr_i32
; CHECK:       orr w0, w0, w1
  %1 = or i32 %a, %b
  ret i32 %1
}

define i64 @or_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: or_rr_i64
; CHECK:       orr x0, x0, x1
  %1 = or i64 %a, %b
  ret i64 %1
}

define i32 @or_ri_i32(i32 %a) {
; CHECK-LABEL: or_ri_i32
; CHECK:       orr w0, w0, #0xff
  %1 = or i32 %a, 255
  ret i32 %1
}

define i64 @or_ri_i64(i64 %a) {
; CHECK-LABEL: or_ri_i64
; CHECK:       orr x0, x0, #0xff
  %1 = or i64 %a, 255
  ret i64 %1
}

define i32 @or_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: or_rs_i32
; CHECK:       orr w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = or i32 %a, %1
  ret i32 %2
}

define i64 @or_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: or_rs_i64
; CHECK:       orr x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = or i64 %a, %1
  ret i64 %2
}

; XOR
define i32 @xor_rr_i32(i32 %a, i32 %b) {
; CHECK-LABEL: xor_rr_i32
; CHECK:       eor w0, w0, w1
  %1 = xor i32 %a, %b
  ret i32 %1
}

define i64 @xor_rr_i64(i64 %a, i64 %b) {
; CHECK-LABEL: xor_rr_i64
; CHECK:       eor x0, x0, x1
  %1 = xor i64 %a, %b
  ret i64 %1
}

define i32 @xor_ri_i32(i32 %a) {
; CHECK-LABEL: xor_ri_i32
; CHECK:       eor w0, w0, #0xff
  %1 = xor i32 %a, 255
  ret i32 %1
}

define i64 @xor_ri_i64(i64 %a) {
; CHECK-LABEL: xor_ri_i64
; CHECK:       eor x0, x0, #0xff
  %1 = xor i64 %a, 255
  ret i64 %1
}

define i32 @xor_rs_i32(i32 %a, i32 %b) {
; CHECK-LABEL: xor_rs_i32
; CHECK:       eor w0, w0, w1, lsl #8
  %1 = shl i32 %b, 8
  %2 = xor i32 %a, %1
  ret i32 %2
}

define i64 @xor_rs_i64(i64 %a, i64 %b) {
; CHECK-LABEL: xor_rs_i64
; CHECK:       eor x0, x0, x1, lsl #8
  %1 = shl i64 %b, 8
  %2 = xor i64 %a, %1
  ret i64 %2
}

