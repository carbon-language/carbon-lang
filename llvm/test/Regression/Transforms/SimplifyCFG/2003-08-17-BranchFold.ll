; This test checks to make sure that 'br X, Dest, Dest' is folded into 
; 'br Dest'

; RUN: as < %s | opt -simplifycfg | dis | not grep 'br bool %c2'

declare void %noop()

int %test(bool %c1, bool %c2) {
	call void %noop()
	br bool %c1, label %A, label %Y
A:
	call void %noop()
 	br bool %c2, label %X, label %X   ; Can be converted to unconditional br
X:
	call void %noop()
	ret int 0
Y:
	call void %noop()
	br label %X
}
