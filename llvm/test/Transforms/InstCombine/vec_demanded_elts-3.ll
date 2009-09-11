; RUN: opt < %s -instcombine -S | not grep load
; PR4340

define void @vac(<4 x float>* nocapture %a) nounwind {
entry:
	%tmp1 = load <4 x float>* %a		; <<4 x float>> [#uses=1]
	%vecins = insertelement <4 x float> %tmp1, float 0.000000e+00, i32 0	; <<4 x float>> [#uses=1]
	%vecins4 = insertelement <4 x float> %vecins, float 0.000000e+00, i32 1; <<4 x float>> [#uses=1]
	%vecins6 = insertelement <4 x float> %vecins4, float 0.000000e+00, i32 2; <<4 x float>> [#uses=1]
	%vecins8 = insertelement <4 x float> %vecins6, float 0.000000e+00, i32 3; <<4 x float>> [#uses=1]
	store <4 x float> %vecins8, <4 x float>* %a
	ret void
}

