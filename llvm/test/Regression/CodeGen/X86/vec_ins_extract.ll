; RUN: llvm-as< %s | llc -march=x86 -mcpu=yonah &&
; RUN: llvm-as< %s | llc -march=x86 -mcpu=yonah | not grep sub.*esp

; This checks that various insert/extract idiom work without going to the 
; stack.
; XFAIL: *

void %test(<4 x float>* %F, float %f) {
entry:
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp3, float %f, uint 0		; <<4 x float>> [#uses=2]
	%tmp6 = add <4 x float> %tmp10, %tmp10		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
}

void %test2(<4 x float>* %F, float %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=3]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp = getelementptr <4 x float>* %G, int 0, int 2		; <float*> [#uses=1]
	store float %f, float* %tmp
	%tmp4 = load <4 x float>* %G		; <<4 x float>> [#uses=2]
	%tmp6 = add <4 x float> %tmp4, %tmp4		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
}

void %test3(<4 x float>* %F, float* %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=2]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp = getelementptr <4 x float>* %G, int 0, int 2		; <float*> [#uses=1]
	%tmp = load float* %tmp		; <float> [#uses=1]
	store float %tmp, float* %f
	ret void
}

void %test4(<4 x float>* %F, float* %f) {
entry:
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp5.lhs = extractelement <4 x float> %tmp, uint 0		; <float> [#uses=1]
	%tmp5.rhs = extractelement <4 x float> %tmp, uint 0		; <float> [#uses=1]
	%tmp5 = add float %tmp5.lhs, %tmp5.rhs		; <float> [#uses=1]
	store float %tmp5, float* %f
	ret void
}
