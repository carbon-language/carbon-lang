; RUN: llvm-as < %s | opt -raise

declare void %foo()

void %test() {
	%X = cast void()* %foo to int()*
	%retval = call int %X()
	ret void
}
