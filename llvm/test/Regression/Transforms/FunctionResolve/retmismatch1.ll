; This shows where the function is called with the prototype indicating a
; return type exists, but it really doesn't.
; RUN: as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.' | not grep call

declare int %foo(...)

void %foo(int %x, float %y) {
	ret void
}

int %bar() {
	%x = call int(...)* %foo(double 12.5, int 48)
	ret int %x
}
