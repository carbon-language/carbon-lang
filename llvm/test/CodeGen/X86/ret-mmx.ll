; RUN: llvm-as < %s | llc -march=x86-64 -mattr=+mmx
; rdar://6602459

@g_v1di = external global <1 x i64>

define void @test_v1di() nounwind {
entry:
	%call = call <1 x i64> @return_v1di()		; <<1 x i64>> [#uses=0]
	store <1 x i64> %call, <1 x i64>* @g_v1di
        ret void
}

declare <1 x i64> @return_v1di()
