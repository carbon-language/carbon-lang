; RUN: opt < %s -instcombine -S | FileCheck %s

; PR22723: Recognize De Morgan's Laws when obfuscated by zexts.

define i32 @demorgan_or(i1 %X, i1 %Y) {
  %zextX = zext i1 %X to i32
  %zextY = zext i1 %Y to i32
  %notX  = xor i32 %zextX, 1
  %notY  = xor i32 %zextY, 1
  %or    = or i32 %notX, %notY
  ret i32 %or

; CHECK-LABEL: demorgan_or(
; CHECK-NEXT:  = zext
; CHECK-NEXT:  = zext
; CHECK-NEXT:  = xor
; CHECK-NEXT:  = xor
; CHECK-NEXT:  = or
; CHECK-NEXT:  ret
}

define i32 @demorgan_and(i1 %X, i1 %Y) {
  %zextX = zext i1 %X to i32
  %zextY = zext i1 %Y to i32
  %notX  = xor i32 %zextX, 1
  %notY  = xor i32 %zextY, 1
  %and   = and i32 %notX, %notY
  ret i32 %and

; CHECK-LABEL: demorgan_and(
; CHECK-NEXT:  = zext
; CHECK-NEXT:  = zext
; CHECK-NEXT:  = xor
; CHECK-NEXT:  = xor
; CHECK-NEXT:  = and
; CHECK-NEXT:  ret
}

