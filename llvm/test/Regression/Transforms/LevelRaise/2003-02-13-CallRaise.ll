; RUN: llvm-upgrade < %s | llvm-as | opt -raise

declare void %foo()

void %test() {
	%X = cast void()* %foo to int()*
	%retval = call int %X()
	ret void
}
