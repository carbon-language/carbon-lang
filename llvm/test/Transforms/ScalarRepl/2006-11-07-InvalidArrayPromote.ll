; RUN: llvm-upgrade < %s | llvm-as | opt -scalarrepl | llvm-dis | \
; RUN:   grep -F {alloca \[2 x <4 x i32>\]}

int %func(<4 x float> %v0, <4 x float> %v1) {
	%vsiidx = alloca [2 x <4 x int>], align 16		; <[2 x <4 x int>]*> [#uses=3]
	%tmp = call <4 x int> %llvm.x86.sse2.cvttps2dq( <4 x float> %v0 )		; <<4 x int>> [#uses=2]
	%tmp = cast <4 x int> %tmp to <2 x long>		; <<2 x long>> [#uses=0]
	%tmp = getelementptr [2 x <4 x int>]* %vsiidx, int 0, int 0		; <<4 x int>*> [#uses=1]
	store <4 x int> %tmp, <4 x int>* %tmp
	%tmp10 = call <4 x int> %llvm.x86.sse2.cvttps2dq( <4 x float> %v1 )		; <<4 x int>> [#uses=2]
	%tmp10 = cast <4 x int> %tmp10 to <2 x long>		; <<2 x long>> [#uses=0]
	%tmp14 = getelementptr [2 x <4 x int>]* %vsiidx, int 0, int 1		; <<4 x int>*> [#uses=1]
	store <4 x int> %tmp10, <4 x int>* %tmp14
	%tmp15 = getelementptr [2 x <4 x int>]* %vsiidx, int 0, int 0, int 4		; <int*> [#uses=1]
	%tmp = load int* %tmp15		; <int> [#uses=1]
	ret int %tmp
}

declare <4 x int> %llvm.x86.sse2.cvttps2dq(<4 x float>)
