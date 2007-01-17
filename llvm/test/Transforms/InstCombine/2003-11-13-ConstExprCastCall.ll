; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep call | notcast

declare void %free(sbyte*)

void %test(int* %X) {
	call int (...)* cast (void (sbyte*)* %free to int (...)*)(int * %X)
	ret void
}

