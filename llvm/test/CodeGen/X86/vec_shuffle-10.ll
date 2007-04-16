; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 | \
; RUN:   grep unpcklps | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 | \
; RUN:   grep unpckhps | wc -l | grep 1
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 | \
; RUN:   not grep {sub.*esp}

void %test(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B) {
        %tmp = load <4 x float>* %B             ; <<4 x float>> [#uses=2]
        %tmp3 = load <4 x float>* %A            ; <<4 x float>> [#uses=2]
        %tmp = extractelement <4 x float> %tmp3, uint 0         ; <float> [#uses=1]
        %tmp7 = extractelement <4 x float> %tmp, uint 0         ; <float> [#uses=1]
        %tmp8 = extractelement <4 x float> %tmp3, uint 1                ; <float> [#uses=1]
        %tmp9 = extractelement <4 x float> %tmp, uint 1         ; <float> [#uses=1]
        %tmp10 = insertelement <4 x float> undef, float %tmp, uint 0            ; <<4 x float>> [#uses=1]
        %tmp11 = insertelement <4 x float> %tmp10, float %tmp7, uint 1          ; <<4 x float>> [#uses=1]
        %tmp12 = insertelement <4 x float> %tmp11, float %tmp8, uint 2          ; <<4 x float>> [#uses=1]
        %tmp13 = insertelement <4 x float> %tmp12, float %tmp9, uint 3          ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp13, <4 x float>* %res
        ret void
}

void %test2(<4 x float> %X, <4 x float>* %res) {
	%tmp5 = shufflevector <4 x float> %X, <4 x float> undef, <4 x uint> < uint 2, uint 6, uint 3, uint 7 >
	store <4 x float> %tmp5, <4 x float>* %res
	ret void
}
