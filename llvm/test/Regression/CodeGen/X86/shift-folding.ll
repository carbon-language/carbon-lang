; RUN: llvm-as < %s | llc -march=x86 | grep 's[ah][rl]l' | wc -l | grep 1

int* %test1(int *%P, uint %X) {
	%Y = shr uint %X, ubyte 2
	%P2 = getelementptr int* %P, uint %Y
	ret int* %P2
}

int* %test2(int *%P, uint %X) {
	%Y = shl uint %X, ubyte 2
	%P2 = getelementptr int* %P, uint %Y
	ret int* %P2
}

int* %test3(int *%P, int %X) {
	%Y = shr int %X, ubyte 2
	%P2 = getelementptr int* %P, int %Y
	ret int* %P2
}
