; This test checks to make sure that 'br X, Dest, Dest' is folded into 
; 'br Dest'.  This can only happen after the 'Z' block is eliminated.  This is
; due to the fact that the SimplifyCFG function does not use 
; the ConstantFoldTerminator function.

; RUN: as < %s | opt -simplifycfg | dis | not grep 'br bool %c2'

declare void %noop()

int %test(bool %c1, bool %c2) {
	call void %noop()
	br bool %c1, label %A, label %Y
A:
	call void %noop()
 	br bool %c2, label %Z, label %X   ; Can be converted to unconditional br
Z:
	br label %X
X:
	call void %noop()
	ret int 0
Y:
	call void %noop()
	br label %X
}
