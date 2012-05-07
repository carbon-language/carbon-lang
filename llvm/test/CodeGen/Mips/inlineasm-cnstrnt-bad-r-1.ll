;
; Register constraint "r" shouldn't take long long unless
; The target is 64 bit.
;
; RUN: not llc -march=mipsel < %s  2> %t
; RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

define i32 @main() nounwind {
entry:

; r with long long
;CHECK-ERRORS:	error: couldn't allocate output register for constraint 'r'

  tail call i64 asm sideeffect "addi $0,$1,$2", "=r,r,i"(i64 7, i64 3) nounwind
  ret i32 0
}

