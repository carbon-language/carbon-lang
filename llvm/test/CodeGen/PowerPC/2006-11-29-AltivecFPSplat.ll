; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mcpu=g5

void %glgRunProcessor15() {
	%tmp26355.i = shufflevector <4 x float> zeroinitializer, <4 x float> < float 0x379FFFE000000000, float 0x379FFFE000000000, float 0x379FFFE000000000, float 0x379FFFE000000000 >, <4 x uint> < uint 0, uint 1, uint 2, uint 7 >		; <<4 x float>> [#uses=1]
	%tmp3030030304.i = cast <4 x float> %tmp26355.i to <8 x short>		; <<8 x short>> [#uses=1]
	%tmp30305.i = shufflevector <8 x short> zeroinitializer, <8 x short> %tmp3030030304.i, <8 x uint> < uint 1, uint 3, uint 5, uint 7, uint 9, uint 11, uint 13, uint 15 >		; <<8 x short>> [#uses=1]
	%tmp30305.i = cast <8 x short> %tmp30305.i to <4 x int>		; <<4 x int>> [#uses=1]
	store <4 x int> %tmp30305.i, <4 x int>* null
	ret void
}
