; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep lwz
int %test(int* %P) {
	store int 1, int* %P
	%V = load int* %P
	ret int %V
}
