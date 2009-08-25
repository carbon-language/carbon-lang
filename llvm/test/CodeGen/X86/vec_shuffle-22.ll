; RUN: llvm-as < %s | llc -march=x86 -mcpu=pentium-m -o %t
; RUN: grep movlhps %t | count 1
; RUN: grep pshufd %t | count 1
; RUN: llvm-as < %s | llc -march=x86 -mcpu=core2 -o %t
; RUN: grep movlhps %t | count 1
; RUN: grep movddup %t | count 1

define <4 x float> @t1(<4 x float> %a) nounwind  {
entry:
        %tmp1 = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >       ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp1
}

define <4 x i32> @t2(<4 x i32>* %a) nounwind {
entry:
        %tmp1 = load <4 x i32>* %a;
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> < i32 0, i32 1, i32 0, i32 1 >		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp2
}
