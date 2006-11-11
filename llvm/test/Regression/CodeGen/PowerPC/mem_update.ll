; RUN: llvm-as < %s | llc -march=ppc32 -enable-ppc-preinc &&
; RUN: llvm-as < %s | llc -march=ppc32 -enable-ppc-preinc | not grep addi &&
; RUN: llvm-as < %s | llc -march=ppc64 -enable-ppc-preinc &&
; RUN: llvm-as < %s | llc -march=ppc64 -enable-ppc-preinc | not grep addi

int *%test0(int *%X,  int *%dest) {
	%Y = getelementptr int* %X, int 4
	%A = load int* %Y
	store int %A, int* %dest
	ret int* %Y
}

int *%test1(int *%X,  int *%dest) {
	%Y = getelementptr int* %X, int 4
	%A = load int* %Y
	store int %A, int* %dest
	ret int* %Y
}

short *%test2(short *%X, int *%dest) {
	%Y = getelementptr short* %X, int 4
	%A = load short* %Y
	%B = cast short %A to int
	store int %B, int* %dest
	ret short* %Y
}

ushort *%test3(ushort *%X, int *%dest) {
	%Y = getelementptr ushort* %X, int 4
	%A = load ushort* %Y
	%B = cast ushort %A to int
	store int %B, int* %dest
	ret ushort* %Y
}


long *%test4(long *%X, long *%dest) {
	%Y = getelementptr long* %X, int 4
	%A = load long* %Y
	store long %A, long* %dest
	ret long* %Y
}
