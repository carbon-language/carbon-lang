; RUN: llvm-as < %s | opt -dse -scalarrepl -instcombine | llvm-dis | not grep 'ret int undef'

int %test(double %__x) {
	%__u = alloca { [3 x int] }
	%tmp.1 = cast { [3 x int] }* %__u to double*
	store double %__x, double* %tmp.1
	%tmp.4 = getelementptr { [3 x int] }* %__u, int 0, uint 0, int 1
	%tmp.5 = load int* %tmp.4
	%tmp.6 = setlt int %tmp.5, 0
	%tmp.7 = cast bool %tmp.6 to int
	ret int %tmp.7
}

