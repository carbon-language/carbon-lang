; RUN: llc -march=mipsel < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; First I with short
; CHECK: #APP
; CHECK: addi $3,$2,4096
; CHECK: #NO_APP
  tail call i16 asm sideeffect "addi $0,$1,$2", "=r,r,I"(i16 7, i16 4096) nounwind

; Then I with int
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},-3
; CHECK: #NO_APP
   tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,I"(i32 7, i32 -3) nounwind

  ret i32 0
}

