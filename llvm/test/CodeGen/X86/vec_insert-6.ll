; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep pslldq

define <4 x float> @t3(<4 x float>* %P) nounwind  {
	%tmp1 = load <4 x float>* %P
	%tmp2 = shufflevector <4 x float> zeroinitializer, <4 x float> %tmp1, <4 x i32> < i32 4, i32 4, i32 4, i32 0 >
	ret <4 x float> %tmp2
}
