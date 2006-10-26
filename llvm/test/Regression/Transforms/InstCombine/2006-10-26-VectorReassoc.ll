; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep mul | wc -l | grep 2


<4 x float> %test(<4 x float> %V) {
	%Y = mul <4 x float> %V, <float 1.0, float 2.0, float 3.0, float 4.0>
	%Z = mul <4 x float> %Y, <float 1.0, float 200000.0, float -3.0, float 4.0>
	ret <4 x float> %Z
}
