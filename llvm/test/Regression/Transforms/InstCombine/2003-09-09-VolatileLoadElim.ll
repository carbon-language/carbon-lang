; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep load
void %test(int* %P) {
	%X = volatile load int* %P  ; Dead but not deletable!
	ret void
}
