; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep shl &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep cast

; This cannot be turned into a sign extending cast!

long %test(long %X) {
	%Y = shl long %X, ubyte 16
	%Z = shr long %Y, ubyte 16
	ret long %Z
}
