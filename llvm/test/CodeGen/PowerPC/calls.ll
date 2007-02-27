; Test various forms of calls.

; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'bl ' | wc -l | grep 2 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'bctrl' | wc -l | grep 1 &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep 'bla ' | wc -l | grep 1
; XFAIL: [0-9x]86-pc-linux.*

declare void %foo()

void %test_direct() {
	call void %foo()
	ret void
}

void %test_extsym(sbyte *%P) {
	free sbyte* %P
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
