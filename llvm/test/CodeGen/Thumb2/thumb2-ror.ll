; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s


; CHECK: f1:
; CHECK: 	ror.w	r0, r0, #22
define i32 @f1(i32 %a) {
    %l8 = shl i32 %a, 10
    %r8 = lshr i32 %a, 22
    %tmp = or i32 %l8, %r8
    ret i32 %tmp
}

; CHECK: f2:
; CHECK: ror
define i32 @f2(i32 %v, i32 %nbits) {
entry:
  %shr = lshr i32 %v, %nbits
  %sub = sub i32 32, %nbits
  %shl = shl i32 %v, %sub
  %or = or i32 %shl, %shr
  ret i32 %or
}