; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | grep 'load int\* %A'

implementation

internal int %callee(bool %C, int* %P) {
	br bool %C, label %T, label %F
T:
	ret int 17
F:
	%X = load int* %P
	ret int %X
}

int %foo() {
	%A = alloca int
	store int 17, int* %A
	%X = call int %callee(bool false, int* %A)
	ret int %X
}

