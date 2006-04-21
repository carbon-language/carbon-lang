; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movss | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movd  | wc -l | grep 1

<4 x float> %test1(float %a) {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, uint 0
	%tmp5 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1
	%tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, uint 2
	%tmp7 = insertelement <4 x float> %tmp6, float 0.000000e+00, uint 3
	ret <4 x float> %tmp7
}

<2 x long> %test(short %a) {
	%tmp = insertelement <8 x short> zeroinitializer, short %a, uint 0
	%tmp6 = insertelement <8 x short> %tmp, short 0, uint 1
	%tmp8 = insertelement <8 x short> %tmp6, short 0, uint 2
	%tmp10 = insertelement <8 x short> %tmp8, short 0, uint 3
	%tmp12 = insertelement <8 x short> %tmp10, short 0, uint 4
	%tmp14 = insertelement <8 x short> %tmp12, short 0, uint 5
	%tmp16 = insertelement <8 x short> %tmp14, short 0, uint 6
	%tmp18 = insertelement <8 x short> %tmp16, short 0, uint 7
	%tmp19 = cast <8 x short> %tmp18 to <2 x long>
	ret <2 x long> %tmp19
}
