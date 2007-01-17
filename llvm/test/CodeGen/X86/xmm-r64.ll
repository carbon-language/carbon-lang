; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86-64

<4 x int> %test() {
	%tmp1039 = call <4 x int> %llvm.x86.sse2.psll.d( <4 x int> zeroinitializer, <4 x int> zeroinitializer )		; <<4 x int>> [#uses=1]
	%tmp1040 = cast <4 x int> %tmp1039 to <2 x long>		; <<2 x long>> [#uses=1]
	%tmp1048 = add <2 x long> %tmp1040, zeroinitializer		; <<2 x long>> [#uses=1]
	%tmp1048 = cast <2 x long> %tmp1048 to <4 x int>		; <<4 x int>> [#uses=1]
	ret <4 x int>  %tmp1048
}

declare <4 x int> %llvm.x86.sse2.psll.d(<4 x int>, <4 x int>)
