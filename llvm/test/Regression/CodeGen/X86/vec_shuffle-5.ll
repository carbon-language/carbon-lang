; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd | wc -l | grep 1

void %test() {
	%tmp1 = load <4 x float>* null
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x uint> < uint 0, uint 1, uint 6, uint 7 >
	%tmp3 = shufflevector <4 x float> %tmp1, <4 x float> zeroinitializer, <4 x uint> < uint 2, uint 3, uint 6, uint 7 >
	%tmp4 = add <4 x float> %tmp2, %tmp3
	store <4 x float> %tmp4, <4 x float>* null
	ret void
}

