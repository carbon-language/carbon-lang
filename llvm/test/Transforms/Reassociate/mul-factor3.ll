; This should be one add and two multiplies.

; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   opt -reassociate -instcombine | llvm-dis > %t 
; RUN: grep mul %t | wc -l | grep 2
; RUN: grep add %t | wc -l | grep 1

int %test(int %A, int %B, int %C) {
	%aa = mul int %A, %A
	%aab = mul int %aa, %B

	%ac = mul int %A, %C
	%aac = mul int %ac, %A
	%r = add int %aab, %aac
	ret int %r
}
