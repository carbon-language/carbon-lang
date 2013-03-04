; RUN: llc -march=mipsel < %s | FileCheck %s

define i32 @main() nounwind {
entry:

; First I with short
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},4096
; CHECK: #NO_APP
  tail call i16 asm sideeffect "addi $0,$1,$2", "=r,r,I"(i16 7, i16 4096) nounwind

; Then I with int
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},-3
; CHECK: #NO_APP
   tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,I"(i32 7, i32 -3) nounwind

; Now J with 0
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},0
; CHECK: #NO_APP
  tail call i32 asm sideeffect "addi $0,$1,$2\0A\09 ", "=r,r,J"(i32 7, i16 0) nounwind

; Now K with 64
; CHECK: #APP
; CHECK: addu ${{[0-9]+}},${{[0-9]+}},64
; CHECK: #NO_APP	
  tail call i16 asm sideeffect "addu $0,$1,$2\0A\09 ", "=r,r,K"(i16 7, i16 64) nounwind

; Now L with 0x00100000
; CHECK: #APP
; CHECK: add ${{[0-9]+}},${{[0-9]+}},${{[0-9]+}}
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "add $0,$1,$3\0A\09", "=r,r,L,r"(i32 7, i32 1048576, i32 0) nounwind

; Now N with -3
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},-3
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,N"(i32 7, i32 -3) nounwind

; Now O with -3
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},-3
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,O"(i32 7, i16 -3) nounwind

; Now P with 65535
; CHECK: #APP
; CHECK: addi ${{[0-9]+}},${{[0-9]+}},65535
; CHECK: #NO_APP	
  tail call i32 asm sideeffect "addi $0,$1,$2", "=r,r,P"(i32 7, i32 65535) nounwind

; Now R Which takes the address of c
  %c = alloca i32, align 4
  store i32 -4469539, i32* %c, align 4
  %8 = call i32 asm sideeffect "lwl $0, 1 + $1\0A\09lwr $0, 2 + $1\0A\09", "=r,*R"(i32* %c) #1
; CHECK: #APP
; CHECK: lwl ${{[0-9]+}}, 1 + 0(${{[0-9]+}})
; CHECK: lwr ${{[0-9]+}}, 2 + 0(${{[0-9]+}})
; CHECK: #NO_APP	

  ret i32 0
}
