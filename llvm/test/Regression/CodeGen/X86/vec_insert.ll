; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movss  | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pinsrw | wc -l | grep 2

void %test(<4 x float>* %F, int %I) {
	%tmp = load <4 x float>* %F
	%f = cast int %I to float
	%tmp1 = insertelement <4 x float> %tmp, float %f, uint 0
	%tmp18 = add <4 x float> %tmp1, %tmp1
	store <4 x float> %tmp18, <4 x float>* %F
	ret void
}

void %test2(<4 x float>* %F, int %I, float %g) {
	%tmp = load <4 x float>* %F
	%f = cast int %I to float
	%tmp1 = insertelement <4 x float> %tmp, float %f, uint 2
	store <4 x float> %tmp1, <4 x float>* %F
	ret void
}
