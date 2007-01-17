; RUN: llvm-upgrade < %s | llvm-as | llc -march=c | grep volatile

void %test(int* %P) {
	%X = volatile load int*%P
	volatile store int %X, int* %P
	ret void
}
