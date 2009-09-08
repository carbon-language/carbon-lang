; RUN: llc < %s -march=x86 -mattr=+sse,-sse2 -mtriple=i386-apple-darwin -o %t
; RUN: grep shufps %t | count 4
; RUN: grep movaps %t | count 2
; RUN: llc < %s -march=x86 -mattr=+sse2 -mtriple=i386-apple-darwin -o %t
; RUN: grep pshufd %t | count 4
; RUN: not grep shufps %t
; RUN: not grep mov %t

define <4 x float> @t1(<4 x float> %a, <4 x float> %b) nounwind  {
        %tmp1 = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> zeroinitializer
        ret <4 x float> %tmp1
}

define <4 x float> @t2(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >
	ret <4 x float> %tmp
}

define <4 x float> @t3(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 4, i32 4, i32 4, i32 4 >
	ret <4 x float> %tmp
}

define <4 x float> @t4(<4 x float> %A, <4 x float> %B) nounwind {
	%tmp = shufflevector <4 x float> %A, <4 x float> %B, <4 x i32> < i32 1, i32 3, i32 2, i32 0 >
	ret <4 x float> %tmp
}
