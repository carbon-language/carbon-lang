; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v 'store.*,.*null' | not grep store

void %test1(int* %P) {
	store int undef, int* %P
	store int 123, int* undef
	store int 124, int* null
	ret void
}

void %test2(int* %P) {
	%X = load int* %P
	%Y = add int %X, 0
	store int %Y, int* %P
	ret void
}
