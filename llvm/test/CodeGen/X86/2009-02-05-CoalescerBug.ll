; RUN: llc < %s -march=x86 -mattr=+sse2,-sse41 | grep movss  | count 2
; RUN: llc < %s -march=x86 -mattr=+sse2,-sse41 | grep movaps | count 4

define i1 @t([2 x float]* %y, [2 x float]* %w, i32, [2 x float]* %x.pn59, i32 %smax190, i32 %j.1180, <4 x float> %wu.2179, <4 x float> %wr.2178, <4 x float>* %tmp89.out, <4 x float>* %tmp107.out, i32* %indvar.next218.out) nounwind {
newFuncRoot:
	%tmp82 = insertelement <4 x float> %wr.2178, float 0.000000e+00, i32 0		; <<4 x float>> [#uses=1]
	%tmp85 = insertelement <4 x float> %tmp82, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp87 = insertelement <4 x float> %tmp85, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp89 = insertelement <4 x float> %tmp87, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp89, <4 x float>* %tmp89.out
	ret i1 false
}
