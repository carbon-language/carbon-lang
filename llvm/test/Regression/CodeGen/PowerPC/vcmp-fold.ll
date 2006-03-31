; This should fold the "vcmpbfp." and "vcmpbfp" instructions into a single 
; "vcmpbfp.".
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep vcmpbfp | wc -l | grep 1

void %test(<4 x float>* %x, <4 x float>* %y, int* %P) {
entry:
	%tmp = load <4 x float>* %x		; <<4 x float>> [#uses=1]
	%tmp2 = load <4 x float>* %y		; <<4 x float>> [#uses=1]
	%tmp = call int %llvm.ppc.altivec.vcmpbfp.p( int 1, <4 x float> %tmp, <4 x float> %tmp2 )		; <int> [#uses=1]
	%tmp4 = load <4 x float>* %x		; <<4 x float>> [#uses=1]
	%tmp6 = load <4 x float>* %y		; <<4 x float>> [#uses=1]
	%tmp = call <4 x int> %llvm.ppc.altivec.vcmpbfp( <4 x float> %tmp4, <4 x float> %tmp6 )		; <<4 x int>> [#uses=1]
	%tmp7 = cast <4 x int> %tmp to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp7, <4 x float>* %x
	store int %tmp, int* %P
	ret void
}

declare int %llvm.ppc.altivec.vcmpbfp.p(int, <4 x float>, <4 x float>)

declare <4 x int> %llvm.ppc.altivec.vcmpbfp(<4 x float>, <4 x float>)
