; RUN: llc < %s -march=x86-64 -mattr=+mmx,+sse2
; rdar://6602459

@g_v1di = external global <1 x i64>

define void @t1() nounwind {
entry:
	%call = call <1 x i64> @return_v1di()		; <<1 x i64>> [#uses=0]
	store <1 x i64> %call, <1 x i64>* @g_v1di
        ret void
}

declare <1 x i64> @return_v1di()

define <1 x i64> @t2() nounwind {
	ret <1 x i64> <i64 1>
}

define <2 x i32> @t3() nounwind {
	ret <2 x i32> <i32 1, i32 0>
}

define double @t4() nounwind {
	ret double bitcast (<2 x i32> <i32 1, i32 0> to double)
}

