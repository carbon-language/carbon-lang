; Positive test for inline register constraints
;
; RUN: llc -march=mipsel < %s  | FileCheck %s

define i32 @main() nounwind {
entry:

; X with -3
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},0xfffffffffffffffd
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:X}", "=r,r,I"(i32 7, i32 -3) nounwind

; x with -3
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},0xfffd
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:x}", "=r,r,I"(i32 7, i32 -3) nounwind

; d with -3
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},-3
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:d}", "=r,r,I"(i32 7, i32 -3) nounwind

  ret i32 0
}
