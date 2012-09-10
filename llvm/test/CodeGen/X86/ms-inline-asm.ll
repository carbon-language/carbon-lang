; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @t1() nounwind {
entry:
  %0 = tail call i32 asm sideeffect inteldialect "mov eax, $1\0Amov $0, eax", "=r,r,~{eax},~{dirflag},~{fpsr},~{flags}"(i32 1) nounwind
  ret i32 %0
; CHECK: t1
; CHECK: {{## InlineAsm Start|#APP}}
; CHECK: .intel_syntax
; CHECK: mov eax, ecx
; CHECK: mov ecx, eax
; CHECK: .att_syntax
; CHECK: {{## InlineAsm End|#NO_APP}}
}
