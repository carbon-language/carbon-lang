; RUN: llvm-as < %s | llc
; PR4778
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define signext i8 @__nesc_atomic_start() nounwind {
entry:
  %0 = tail call i16 asm sideeffect "mov    r2, $0", "=r"() nounwind ; <i16> [#uses=1]
  %1 = trunc i16 %0 to i8                         ; <i8> [#uses=1]
  %and3 = lshr i8 %1, 3                           ; <i8> [#uses=1]
  %conv1 = and i8 %and3, 1                        ; <i8> [#uses=1]
  tail call void asm sideeffect "dint", ""() nounwind
  tail call void asm sideeffect "nop", ""() nounwind
  tail call void asm sideeffect "", "~{memory}"() nounwind
  ret i8 %conv1
}
