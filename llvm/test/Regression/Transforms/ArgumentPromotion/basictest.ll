; RUN: llvm-as < %s | opt -argpromotion -mem2reg | llvm-dis | not grep alloca

implementation

internal int %test(int *%X, int* %Y) {
	%A = load int* %X
	%B = load int* %Y
	%C = add int %A, %B
	ret int %C
}

internal int %caller(int* %B) {
	%A = alloca int
	store int 1, int* %A
	%C = call int %test(int* %A, int* %B)
	ret int %C
}

int %callercaller() {
	%B = alloca int
	store int 2, int* %B
	%X = call int %caller(int* %B)
	ret int %X
}
