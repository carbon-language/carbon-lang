; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movss    %t | wc -l | grep 3 
; RUN: grep movhlps  %t | wc -l | grep 1 
; RUN: grep pshufd   %t | wc -l | grep 1 
; RUN: grep unpckhpd %t | wc -l | grep 1

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

void %test3(float* %R, <4 x float>* %P1) {
	%X = load <4 x float>* %P1
	%tmp = extractelement <4 x float> %X, uint 3
	store float %tmp, float* %R
	ret void
}

double %test4(double %A) {
        %tmp1 = call <2 x double> %foo()
        %tmp2 = extractelement <2 x double> %tmp1, uint 1
        %tmp3 = add double %tmp2, %A
        ret double %tmp3
}

declare <2 x double> %foo()
