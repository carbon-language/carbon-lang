; The cast in this testcase is not eliminatable on a 32-bit target!
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep cast

target endian = little
target pointersize = 32

declare void %foo(...)

void %test(long %X) {
	%Y = cast long %X to int*
	call void (...)* %foo(int* %Y)
	ret void
}
