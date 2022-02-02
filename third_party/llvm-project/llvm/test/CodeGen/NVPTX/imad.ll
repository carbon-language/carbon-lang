; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; CHECK: imad
define i32 @imad(i32 %a, i32 %b, i32 %c) {
; CHECK: mad.lo.s32
  %val0 = mul i32 %a, %b
  %val1 = add i32 %val0, %c
  ret i32 %val1
}
