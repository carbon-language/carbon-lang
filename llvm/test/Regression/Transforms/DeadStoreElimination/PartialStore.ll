; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep 'store sbyte'
; Ensure that the dead store is deleted in this case.  It is wholely
; overwritten by the second store.
int %test() {
	%V = alloca int
	%V2 = cast int* %V to sbyte*
	store sbyte 0, sbyte* %V2
	store int 1234567, int* %V
	%X = load int* %V
	ret int %X
}
