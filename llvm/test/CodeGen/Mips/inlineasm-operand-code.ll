; Positive test for inline register constraints
;
; RUN: llc -march=mipsel < %s  | FileCheck %s -check-prefix=LITTLE
; RUN: llc -march=mips < %s  | FileCheck %s -check-prefix=BIG

%union.u_tag = type { i64 }
%struct.anon = type { i32, i32 }
@uval = common global %union.u_tag zeroinitializer, align 8
define i32 @main() nounwind {
entry:

; X with -3
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},0xfffffffffffffffd
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:X}", "=r,r,I"(i32 7, i32 -3) nounwind

; x with -3
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},0xfffd
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:x}", "=r,r,I"(i32 7, i32 -3) nounwind

; d with -3
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},-3
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:d}", "=r,r,I"(i32 7, i32 -3) nounwind

; m with -3
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},-4
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:m}", "=r,r,I"(i32 7, i32 -3) nounwind

; z with -3
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},-3
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 -3) nounwind

; z with 0
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},$0
;LITTLE:	#NO_APP
  tail call i32 asm sideeffect "addi $0,$1,${2:z}", "=r,r,I"(i32 7, i32 0) nounwind

; a long long in 32 bit mode (use to assert)
;LITTLE:	#APP
;LITTLE:	addi ${{[0-9]+}},${{[0-9]+}},3
;LITTLE:	#NO_APP
  tail call i64 asm sideeffect "addi $0,$1,$2 \0A\09", "=r,r,X"(i64 1229801703532086340, i64 3) nounwind

; D, in little endian the source reg will be 4 bytes into the long long
;LITTLE:    lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;LITTLE:    lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;LITTLE-NEXT: lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;LITTLE:	#APP
;LITTLE:    or	${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;LITTLE:    #NO_APP

; D, in big endian the source reg will also be 4 bytes into the long long
;BIG:       #APP
;BIG:       #APP
;BIG:       #APP
;BIG:       #APP
;BIG:       #APP
;BIG:       #APP
;BIG:       #APP
;BIG:       lw ${{[0-9]+}}, %got(uval)(${{[0-9,a-z]+}})
;BIG:       lw $[[SECOND:[0-9]+]], 4(${{[0-9]+}})
;BIG-NEXT:  lw $[[FIRST:[0-9]+]], 0(${{[0-9]+}})
;BIG:       #APP
;BIG:       or	${{[0-9]+}},$[[SECOND]],${{[0-9]+}}
;BIG:       #NO_APP
  %7 = load i64* getelementptr inbounds (%union.u_tag* @uval, i32 0, i32 0), align 8
  %trunc1 = trunc i64 %7 to i32
  tail call i32 asm sideeffect "or $0,${1:D},$2", "=r,r,r"(i64 %7, i32 %trunc1) nounwind

  ret i32 0
}

