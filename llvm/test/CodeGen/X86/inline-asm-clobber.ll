; RUN: llc <%s -mtriple=x86_64-unknown-unknown -- 2>&1  | FileCheck %s

; CHECK: warning: inline asm clobber list contains reserved registers: RSP, EBP

define void @foo() nounwind {
  call void asm sideeffect "mov $$0x12, %eax", "~{eax},~{rsp},~{esi},~{ebp}"()
  ret void
}
