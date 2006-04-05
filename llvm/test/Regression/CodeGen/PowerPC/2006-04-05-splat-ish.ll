; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep 'vspltish v.*, 10'

void %test(<8 x short>* %P) {
	%tmp = load <8 x short>* %P		; <<8 x short>> [#uses=1]
	%tmp1 = add <8 x short> %tmp, < short 10, short 10, short 10, short 10, short 10, short 10, short 10, short 10 >		; <<8 x short>> [#uses=1]
	store <8 x short> %tmp1, <8 x short>* %P
	ret void
}
