; RUN: llvm-as < %s | opt -argpromotion -instcombine | llvm-dis | not grep load

%G1 = constant int 0
%G2 = constant int* %G1

implementation

internal int %test(int **%X) {
	%Y = load int** %X
	%X = load int* %Y
	ret int %X
}

int %caller(int** %P) {
	%X = call int %test(int** %G2)
	ret int %X
}
