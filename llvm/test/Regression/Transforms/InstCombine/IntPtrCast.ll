; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep cast
target pointersize = 32

int *%test(int *%P) {
	%V = cast int* %P to int
	%P2 = cast int %V to int*
	ret int* %P2
}
