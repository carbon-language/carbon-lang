; RUN: llvm-as < %s | opt -instcombine -disable-output

float %test(<4 x float> %V) {
	%V2 = insertelement <4 x float> %V, float 1.0, uint 3
	%R = extractelement <4 x float> %V2, uint 2
	ret float %R
}
