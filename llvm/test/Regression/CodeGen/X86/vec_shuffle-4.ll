; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shuf | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep unpck
void %test(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B, <4 x float>* %C) {
	%tmp3 = load <4 x float>* %B
	%tmp5 = load <4 x float>* %C
	%tmp11 = shufflevector <4 x float> %tmp3, <4 x float> %tmp5, <4 x uint> < uint 1, uint 4, uint 1, uint 5 >
	store <4 x float> %tmp11, <4 x float>* %res
	ret void
}
