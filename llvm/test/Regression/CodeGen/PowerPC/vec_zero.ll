; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vxor

void %foo(<4 x float> *%P) {
	%T = load <4 x float> * %P
	%S = add <4 x float> zeroinitializer, %T
	store <4 x float> %S, <4 x float>* %P
	ret void
}
