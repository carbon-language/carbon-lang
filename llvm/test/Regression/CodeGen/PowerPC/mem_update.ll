; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep addi

; XFAIL: *

int *%test(int *%X,  int *%dest) {
	%Y = getelementptr int* %X, int 4
	%A = load int* %Y
	store int %A, int* %dest
	ret int* %Y
}
