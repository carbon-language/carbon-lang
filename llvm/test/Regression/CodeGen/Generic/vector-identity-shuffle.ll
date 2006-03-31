; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | grep _test &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep vperm

void %test(<4 x float> *%tmp2.i) {
	%tmp2.i = load <4x float>* %tmp2.i
       %xFloat0.48 = extractelement <4 x float> %tmp2.i, uint 0                ; <float> [#uses=1]
        %inFloat0.49 = insertelement <4 x float> undef, float %xFloat0.48, uint 0               ; <<4 x float>> [#uses=1]
        %xFloat1.50 = extractelement <4 x float> %tmp2.i, uint 1                ; <float> [#uses=1]
        %inFloat1.52 = insertelement <4 x float> %inFloat0.49, float %xFloat1.50, uint 1                ; <<4 x float>> [#uses=1]
        %xFloat2.53 = extractelement <4 x float> %tmp2.i, uint 2                ; <float> [#uses=1]
        %inFloat2.55 = insertelement <4 x float> %inFloat1.52, float %xFloat2.53, uint 2                ; <<4 x float>> [#uses=1]
        %xFloat3.56 = extractelement <4 x float> %tmp2.i, uint 3                ; <float> [#uses=1]
        %inFloat3.58 = insertelement <4 x float> %inFloat2.55, float %xFloat3.56, uint 3                ; <<4 x float>> [#uses=4]
	store <4 x float> %inFloat3.58, <4x float>* %tmp2.i
	ret void
}
