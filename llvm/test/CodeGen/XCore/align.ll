; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK: .align 4
; CHECK-LABEL: f:
define void @f() nounwind {
entry:
  ret void
}

; CHECK: .align 2
; CHECK-LABEL: g:
define void @g() nounwind optsize {
entry:
  ret void
}
