; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep shufflevec | wc -l | grep 1 &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep insertelement &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep extractelement

<4 x float> %test(<4 x float> %tmp, <4 x float> %tmp1) {
	%tmp4 = extractelement <4 x float> %tmp, uint 1		; <float> [#uses=1]
	%tmp2 = extractelement <4 x float> %tmp, uint 3		; <float> [#uses=1]
	%tmp1 = extractelement <4 x float> %tmp1, uint 0		; <float> [#uses=1]
	%tmp128 = insertelement <4 x float> undef, float %tmp4, uint 0		; <<4 x float>> [#uses=1]
	%tmp130 = insertelement <4 x float> %tmp128, float undef, uint 1		; <<4 x float>> [#uses=1]
	%tmp132 = insertelement <4 x float> %tmp130, float %tmp2, uint 2		; <<4 x float>> [#uses=1]
	%tmp134 = insertelement <4 x float> %tmp132, float %tmp1, uint 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp134
}
