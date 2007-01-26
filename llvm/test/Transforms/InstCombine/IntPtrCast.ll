; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | notcast
target endian = little
target pointersize = 32

int *%test(int *%P) {
	%V = cast int* %P to int
	%P2 = cast int %V to int*
	ret int* %P2
}
