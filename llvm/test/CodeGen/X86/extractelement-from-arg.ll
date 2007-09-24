; RUN: llvm-as %s -o - | llc -march=x86-64

define void @test(float* %R, <4 x float> %X) {
	%tmp = extractelement <4 x float> %X, i32 3
	store float %tmp, float* %R
	ret void
}
