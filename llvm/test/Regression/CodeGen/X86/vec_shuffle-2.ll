; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufhw | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshuflw | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movhps | wc -l | grep 1

void %test1(<2 x long>* %res, <2 x long>* %A) {
	%tmp = load <2 x long>* %A
	%tmp = cast <2 x long> %tmp to <8 x short>
	%tmp0 = extractelement <8 x short> %tmp, uint 0
	%tmp1 = extractelement <8 x short> %tmp, uint 1
	%tmp2 = extractelement <8 x short> %tmp, uint 2
	%tmp3 = extractelement <8 x short> %tmp, uint 3
	%tmp4 = extractelement <8 x short> %tmp, uint 4
	%tmp5 = extractelement <8 x short> %tmp, uint 5
	%tmp6 = extractelement <8 x short> %tmp, uint 6
	%tmp7 = extractelement <8 x short> %tmp, uint 7
	%tmp8 = insertelement <8 x short> undef, short %tmp2, uint 0
	%tmp9 = insertelement <8 x short> %tmp8, short %tmp1, uint 1
	%tmp10 = insertelement <8 x short> %tmp9, short %tmp0, uint 2
	%tmp11 = insertelement <8 x short> %tmp10, short %tmp3, uint 3
	%tmp12 = insertelement <8 x short> %tmp11, short %tmp6, uint 4
	%tmp13 = insertelement <8 x short> %tmp12, short %tmp5, uint 5
	%tmp14 = insertelement <8 x short> %tmp13, short %tmp4, uint 6
	%tmp15 = insertelement <8 x short> %tmp14, short %tmp7, uint 7
	%tmp15 = cast <8 x short> %tmp15 to <2 x long>
	store <2 x long> %tmp15, <2 x long>* %res
	ret void
}

void %test2(<4 x float>* %r, <2 x int>* %A) {
	%tmp = load <4 x float>* %r
	%tmp = cast <2 x int>* %A to double*
	%tmp = load double* %tmp
	%tmp = insertelement <2 x double> undef, double %tmp, uint 0
	%tmp5 = insertelement <2 x double> %tmp, double undef, uint 1
	%tmp6 = cast <2 x double> %tmp5 to <4 x float>
	%tmp = extractelement <4 x float> %tmp, uint 0
	%tmp7 = extractelement <4 x float> %tmp, uint 1
	%tmp8 = extractelement <4 x float> %tmp6, uint 0
	%tmp9 = extractelement <4 x float> %tmp6, uint 1
	%tmp10 = insertelement <4 x float> undef, float %tmp, uint 0
	%tmp11 = insertelement <4 x float> %tmp10, float %tmp7, uint 1
	%tmp12 = insertelement <4 x float> %tmp11, float %tmp8, uint 2
	%tmp13 = insertelement <4 x float> %tmp12, float %tmp9, uint 3
	store <4 x float> %tmp13, <4 x float>* %r
	ret void
}
