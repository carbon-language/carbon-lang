; Test various forms of calls.

; RUN: llvm-as < %s | llc -march=ppc32 | grep 'bl ' | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep 'bctrl' | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=ppc32 | grep 'bla ' | wc -l | grep 1

declare void %foo()

void %test_direct() {
	call void %foo()
	ret void
}

void %test_indirect(void()* %fp) {
	call void %fp()
	ret void
}

void %test_abs() {
	%fp = cast int 400 to void()*
	call void %fp()
	ret void
}
