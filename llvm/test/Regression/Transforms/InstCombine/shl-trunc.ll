; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep shl

bool %test(int %X, ubyte %A) {
	%B = lshr int %X, ubyte %A
	%D = trunc int %B to bool
	ret bool %D
}
