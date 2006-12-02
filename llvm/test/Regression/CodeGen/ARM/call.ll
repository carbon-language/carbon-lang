; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep bl &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep blx

void %f() {
entry:
	call void %g( int 1, int 2, int 3, int 4 )
	call fastcc void %h()
	ret void
}

declare void %g(int, int, int, int)
declare fastcc void %h()

void %g(void (...)* %g) {
entry:
	%g_c = cast void (...)* %g to void ()*
	call void %g_c( )
	ret void
}
