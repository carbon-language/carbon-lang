; In this test, a local alloca cannot alias an incoming argument.

; RUN: llvm-as < %s | opt -load-vn -gcse -instcombine | llvm-dis | not grep sub

int %test(int* %P) {
	%X = alloca int
	%V1 = load int* %P
	store int 0, int* %X
	%V2 = load int* %P
	%Diff = sub int %V1, %V2
	ret int %Diff
}
