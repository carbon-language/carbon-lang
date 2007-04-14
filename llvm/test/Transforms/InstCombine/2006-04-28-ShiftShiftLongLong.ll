; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep shl
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | notcast

; This cannot be turned into a sign extending cast!

long %test(long %X) {
	%Y = shl long %X, ubyte 16
	%Z = shr long %Y, ubyte 16
	ret long %Z
}
