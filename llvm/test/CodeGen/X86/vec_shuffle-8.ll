; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 | \
; RUN:   not grep shufps

void %test(<4 x float>* %res, <4 x float>* %A) {
	%tmp1 = load <4 x float>* %A
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x uint> < uint 0, uint 5, uint 6, uint 7 >
	store <4 x float> %tmp2, <4 x float>* %res
	ret void
}
