; This shows where the function is called with the prototype indicating a
; return type doesn't exists, but it really does.
;
; RUN: as < %s | opt -funcresolve -instcombine | dis | grep '\.\.\.' | not grep call

declare void %foo(...)

int %foo(int %x, float %y) {
	ret int %x
}

int %bar() {
	call void (...)* %foo(double 12.5, int 48)
	ret int 6
}
