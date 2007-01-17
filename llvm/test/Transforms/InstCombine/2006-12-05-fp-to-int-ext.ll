; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep zext

; Never merge these two conversions, even though it's possible: this is
; significantly more expensive than the two conversions on some targets
; and it causes libgcc to be compile __fixunsdfdi into a recursive 
; function.


long %test(double %D) {
	%A = fptoui double %D to uint
	%B = zext uint %A to long
	ret long %B
}
