; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK-LABEL: f1
; CHECK: stw lr, sp[0]
; CHECK: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
define void @f1() nounwind {
entry:
  tail call void asm sideeffect "", "~{lr}"() nounwind
  ret void
}

; CHECK-LABEL: f3
; CHECK: entsp 2
; CHECK: stw [[REG:r[4-9]+]], sp[1]
; CHECK: mov [[REG]], r0
; CHECK: bl f2
; CHECK: mov r0, [[REG]]
; CHECK: ldw [[REG]], sp[1]
; CHECK: retsp 2
declare void @f2()
define i32 @f3(i32 %i) nounwind {
entry:
  call void @f2()
  ret i32 %i
}
