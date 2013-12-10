; RUN: not llc -march x86 < %s 2>&1 | FileCheck %s

; We don't currently support realigning the stack and adjusting the stack
; pointer in inline asm.  This commonly happens in MS inline assembly using
; push and pop.

; CHECK: Stack realignment in presence of dynamic stack adjustments is not supported with inline assembly

define i32 @foo() {
entry:
  %r = alloca i32, align 16
  store i32 -1, i32* %r, align 16
  call void asm sideeffect inteldialect "push esi\0A\09xor esi, esi\0A\09mov dword ptr $0, esi\0A\09pop esi", "=*m,~{flags},~{esi},~{esp},~{dirflag},~{fpsr},~{flags}"(i32* %r)
  %0 = load i32* %r, align 16
  ret i32 %0
}
