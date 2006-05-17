; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

<4 x float> %opRSQ(<4 x float> %a) {
entry:
	%tmp2 = extractelement <4 x float> %a, uint 3
	%abscond = setge float %tmp2, -0.000000e+00
	%abs = select bool %abscond, float %tmp2, float 0.000000e+00
	%tmp3 = tail call float %llvm.sqrt.f32( float %abs )
	%tmp4 = div float 1.000000e+00, %tmp3
	%tmp11 = insertelement <4 x float> zeroinitializer, float %tmp4, uint 3
	ret <4 x float> %tmp11
}

declare float %llvm.sqrt.f32(float)
