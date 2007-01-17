; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine

int %test(int %X) {
	%Y = rem int %X, undef
	ret int %Y
}
