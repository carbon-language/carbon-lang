; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

void %foo(bool %C, int* %P) {
	br bool %C, label %T, label %F
T:
	store int 7, int* %P
	ret void
F:
	store int 7, int* %P
	ret void
}
