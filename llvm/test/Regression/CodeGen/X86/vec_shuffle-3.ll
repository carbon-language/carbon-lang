; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlhps | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movhlps | wc -l | grep 1

<4 x float> %test1(<4 x float>* %x, <4 x float>* %y) {
	%tmp = load <4 x float>* %y
	%tmp5 = load <4 x float>* %x
	%tmp9 = add <4 x float> %tmp5, %tmp
	%tmp21 = sub <4 x float> %tmp5, %tmp
	%tmp27 = shufflevector <4 x float> %tmp9, <4 x float> %tmp21, <4 x uint> < uint 0, uint 1, uint 4, uint 5 >
	ret <4 x float> %tmp27
}

<4 x float> %movhl(<4 x float>* %x, <4 x float>* %y) {
entry:
	%tmp = load <4 x float>* %y
	%tmp3 = load <4 x float>* %x
	%tmp4 = shufflevector <4 x float> %tmp3, <4 x float> %tmp, <4 x uint> < uint 2, uint 3, uint 6, uint 7 >
	ret <4 x float> %tmp4
}
