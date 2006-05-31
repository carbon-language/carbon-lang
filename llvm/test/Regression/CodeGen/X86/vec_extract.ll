; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movss    | wc -l | grep 3 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movhlps  | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufd   | wc -l | grep 1

void %test1(<4 x float>* %F, float* %f) {
	%tmp = load <4 x float>* %F
	%tmp7 = add <4 x float> %tmp, %tmp
	%tmp2 = extractelement <4 x float> %tmp7, uint 0
	store float %tmp2, float* %f
	ret void
}

float %test2(<4 x float>* %F, float* %f) {
	%tmp = load <4 x float>* %F
	%tmp7 = add <4 x float> %tmp, %tmp
	%tmp2 = extractelement <4 x float> %tmp7, uint 2
        ret float %tmp2
}

void %test2(float* %R, <4 x float>* %P1) {
	%X = load <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, uint 3
	store float %tmp, float* %R
	ret void
}
