; RUN: as < %s | opt -instcombine | dis | grep load
void %test(int* %P) {
	%X = volatile load int* %P  ; Dead but not deletable!
	ret void
}
