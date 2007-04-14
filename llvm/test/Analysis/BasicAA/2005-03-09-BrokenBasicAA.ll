; RUN: llvm-upgrade < %s | llvm-as | opt -basicaa -load-vn -gcse -instcombine |\
; RUN:    llvm-dis | grep {load i32\\* %A}

declare double* %useit(int*)

int %foo(uint %Amt) {
	%A = malloc int, uint %Amt
	%P = call double*  %useit(int* %A)

	%X = load int* %A
	store double 0.0, double* %P
	%Y = load int* %A
	%Z = sub int %X, %Y
	ret int %Z
}
