; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movups | wc -l | grep 3

%x = global [4 x int] [ int 1, int 2, int 3, int 4 ]

<2 x long> %test1() {
	%tmp = load int* getelementptr ([4 x int]* %x, int 0, int 0)
	%tmp3 = load int* getelementptr ([4 x int]* %x, int 0, int 1)
	%tmp5 = load int* getelementptr ([4 x int]* %x, int 0, int 2)
	%tmp7 = load int* getelementptr ([4 x int]* %x, int 0, int 3)
	%tmp = insertelement <4 x int> undef, int %tmp, uint 0
	%tmp13 = insertelement <4 x int> %tmp, int %tmp3, uint 1
	%tmp14 = insertelement <4 x int> %tmp13, int %tmp5, uint 2
	%tmp15 = insertelement <4 x int> %tmp14, int %tmp7, uint 3
	%tmp16 = cast <4 x int> %tmp15 to <2 x long>
	ret <2 x long> %tmp16
}

<4 x float> %test2(float %a, float %b, float %c, float %d) {
	%tmp = insertelement <4 x float> undef, float %a, uint 0
	%tmp11 = insertelement <4 x float> %tmp, float %b, uint 1
	%tmp12 = insertelement <4 x float> %tmp11, float %c, uint 2
	%tmp13 = insertelement <4 x float> %tmp12, float %d, uint 3
	ret <4 x float> %tmp13
}

<2 x double> %test3(double %a, double %b) {
	%tmp = insertelement <2 x double> undef, double %a, uint 0
	%tmp7 = insertelement <2 x double> %tmp, double %b, uint 1
	ret <2 x double> %tmp7
}
