; RUN: llvm-as < %s | opt -analyze -tddatastructure

void %foo(int* %X) {
	store int 4, int* %X
	ret void
}
