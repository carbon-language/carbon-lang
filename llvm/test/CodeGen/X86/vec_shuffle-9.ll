; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep punpck %t | wc -l | grep 2
; RUN: not grep pextrw %t

<4 x int> %test(sbyte** %ptr) {
entry:
	%tmp = load sbyte** %ptr
	%tmp = cast sbyte* %tmp to float*
	%tmp = load float* %tmp
	%tmp = insertelement <4 x float> undef, float %tmp, uint 0
	%tmp9 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1
	%tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, uint 2
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, uint 3
	%tmp21 = cast <4 x float> %tmp11 to <16 x sbyte>
	%tmp22 = shufflevector <16 x sbyte> %tmp21, <16 x sbyte> zeroinitializer, <16 x uint> < uint 0, uint 16, uint 1, uint 17, uint 2, uint 18, uint 3, uint 19, uint 4, uint 20, uint 5, uint 21, uint 6, uint 22, uint 7, uint 23 >
	%tmp31 = cast <16 x sbyte> %tmp22 to <8 x short>
	%tmp = shufflevector <8 x short> zeroinitializer, <8 x short> %tmp31, <8 x uint> < uint 0, uint 8, uint 1, uint 9, uint 2, uint 10, uint 3, uint 11 >
	%tmp36 = cast <8 x short> %tmp to <4 x int>
	ret <4 x int> %tmp36
}
