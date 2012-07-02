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

; m with -3
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},-4
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:m}", "=r,r,I"(i32 7, i32 -3) nounwind

; z with -3
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},-3
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 -3) nounwind

; z with 0
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},$0
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 0) nounwind

; a long long in 32 bit mode (use to assert)
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},3
;CHECK:	#NO_APP
  tail call i64 asm sideeffect "addi $0,$1,$2 \0A\09", "=r,r,X"(i64 1229801703532086340, i64 3) nounwind

  ret i32 0
}
