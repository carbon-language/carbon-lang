; RUN: llc < %s -mtriple=thumb-apple-darwin -disable-fp-elim | FileCheck %s
; rdar://7268481

define arm_apcscc void @t() nounwind {
; CHECK:       t:
; CHECK-NEXT : push {r7}
entry:
  call void asm sideeffect ".long 0xe7ffdefe", ""() nounwind
  ret void
}
