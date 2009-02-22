; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx
; rdar://6602459

@g_v1di = external global <1 x i64>

define void @t1() nounwind {
entry:
	%call = call <1 x i64> @return_v1di()		; <<1 x i64>> [#uses=0]
	store <1 x i64> %call, <1 x i64>* @g_v1di
        ret void
}

define <1 x i64> @t2() nounwind {
	ret <1 x i64> <i64 1>
}

declare <1 x i64> @return_v1di()
