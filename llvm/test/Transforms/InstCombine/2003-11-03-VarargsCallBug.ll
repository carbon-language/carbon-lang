; The cast in this testcase is not eliminable on a 32-bit target!
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep inttoptr

target endian = little
target pointersize = 32

declare void %foo(...)

void %test(long %X) {
	%Y = cast long %X to int*
	call void (...)* %foo(int* %Y)
	ret void
}
