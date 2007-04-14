; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep -v {store.*,.*null} | not grep store

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
