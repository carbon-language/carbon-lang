; RUN: llvm-as < %s | opt -instcombine

int %test(int %X) {
	%Y = rem int %X, undef
	ret int %Y
}
