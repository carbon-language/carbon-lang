; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v 'store.*,.*null' | not grep store


void %test1(int* %P) {
	store int undef, int* %P
	store int 123, int* undef
	store int 124, int* null
	ret void
}
