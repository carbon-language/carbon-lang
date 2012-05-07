; Positive test for inline register constraints
;
; RUN: llc -march=mipsel < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; r with char
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},23
;CHECK:	#NO_APP
  tail call i8 asm sideeffect "addi $0,$1,$2", "=r,r,n"(i8 27, i8 23) nounwind

; r with short
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},13
;CHECK:	#NO_APP
  tail call i16 asm sideeffect "addi $0,$1,$2", "=r,r,n"(i16 17, i16 13) nounwind

; r with int
;CHECK:	#APP
;CHECK:	addi ${{[0-9]+}},${{[0-9]+}},3
;CHECK:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,n"(i32 7, i32 3) nounwind
 
  ret i32 0
}
