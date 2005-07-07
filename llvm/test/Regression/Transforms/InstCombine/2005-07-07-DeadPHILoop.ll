; RUN: llvm-as < %s | opt -instcombine -disable-output

; This example caused instcombine to spin into an infinite loop.

void %test(int *%P) {
	ret void
Dead:
	%X = phi int [%Y, %Dead]
	%Y = div int %X, 10
	store int %Y, int* %P
	br label %Dead
}

