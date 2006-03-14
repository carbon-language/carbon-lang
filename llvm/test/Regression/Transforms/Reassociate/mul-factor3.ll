; This should be one add and two multiplies.

; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep mul | wc -l | grep 2 &&
; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep add | wc -l | grep 1

int %test(int %A, int %B, int %C) {
	%aa = mul int %A, %A
	%aab = mul int %aa, %B

	%ac = mul int %A, %C
	%aac = mul int %ac, %A
	%r = add int %aab, %aac
	ret int %r
}
