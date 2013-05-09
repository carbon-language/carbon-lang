; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK: f1
; CHECK: stw lr, sp[0]
; CHECK: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
define void @f1() nounwind {
entry:
  tail call void asm sideeffect "", "~{lr}"() nounwind
  ret void
}
