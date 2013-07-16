; RUN: llc < %s -march=xcore | FileCheck %s
; CHECK-LABEL: f1:
; CHECK: foo r0
define i32 @f1() nounwind {
entry:
  %asmtmp = tail call i32 asm sideeffect "foo $0", "=r"() nounwind
  ret i32 %asmtmp
}

; CHECK-LABEL: f2:
; CHECK: foo 5
define void @f2() nounwind {
entry:
  tail call void asm sideeffect "foo $0", "i"(i32 5) nounwind
  ret void
}

; CHECK-LABEL: f3:
; CHECK: foo 42
define void @f3() nounwind {
entry:
  tail call void asm sideeffect "foo ${0:c}", "i"(i32 42) nounwind
  ret void
}

; CHECK-LABEL: f4:
; CHECK: foo -99
define void @f4() nounwind {
entry:
  tail call void asm sideeffect "foo ${0:n}", "i"(i32 99) nounwind
  ret void
}
