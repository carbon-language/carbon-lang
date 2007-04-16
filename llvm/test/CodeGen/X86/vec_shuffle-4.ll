; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 > %t
; RUN: grep shuf %t | wc -l | grep 2
; RUN: not grep unpck %t
void %test(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B, <4 x float>* %C) {
	%tmp3 = load <4 x float>* %B
	%tmp5 = load <4 x float>* %C
	%tmp11 = shufflevector <4 x float> %tmp3, <4 x float> %tmp5, <4 x uint> < uint 1, uint 4, uint 1, uint 5 >
	store <4 x float> %tmp11, <4 x float>* %res
	ret void
}
