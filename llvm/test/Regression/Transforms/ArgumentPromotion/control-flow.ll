; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | not grep 'load int\* null'

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
	%X = call int %callee(bool true, int* null)
	ret int %X
}

