; RUN: llvm-as < %s | opt -load-vn -gcse | llvm-dis | not grep load
; Test that loads of undefined memory are eliminated.

int %test1() {
	%X = malloc int
	%Y = load int* %X
	ret int %Y
}
int %test2() {
	%X = alloca int
	%Y = load int* %X
	ret int %Y
}

