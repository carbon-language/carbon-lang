; RUN: not llvm-as <%s 2>&1  | FileCheck %s

; "~x{21}" is not a valid clobber constraint.

; CHECK: invalid type for inline asm constraint string

define void @foo() nounwind {
  call void asm sideeffect "mov x0, #42", "~{x0},~{x19},~x{21}"() nounwind
  ret void
}
