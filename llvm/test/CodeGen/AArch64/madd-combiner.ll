; RUN: llc -mtriple=aarch64-apple-darwin -verify-machineinstrs < %s | FileCheck %s

; Test that we use the correct register class.
define i32 @mul_add_imm(i32 %a, i32 %b) {
; CHECK-LABEL: mul_add_imm
; CHECK:       orr [[REG:w[0-9]+]], wzr, #0x4
; CHECK-NEXT:  madd  {{w[0-9]+}}, w0, w1, [[REG]]
  %1 = mul i32 %a, %b
  %2 = add i32 %1, 4
  ret i32 %2
}

define i32 @mul_sub_imm1(i32 %a, i32 %b) {
; CHECK-LABEL: mul_sub_imm1
; CHECK:       orr [[REG:w[0-9]+]], wzr, #0x4
; CHECK-NEXT:  msub  {{w[0-9]+}}, w0, w1, [[REG]]
  %1 = mul i32 %a, %b
  %2 = sub i32 4, %1
  ret i32 %2
}
