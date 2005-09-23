; RUN: llvm-as < %s | llc -march=ppc32 | not grep xori &&
; RUN: llvm-as < %s | llc -march=ppc32

int %test(bool %B, int* %P) {
   br bool %B, label %T, label %F
T:
	store int 123, int* %P
	ret int 0
F:
ret int 17
}
