; RUN: llc < %s -march=x86-64
; PR4669
declare <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64>, i32)

define <1 x i64> @test(i64 %t) {
entry:
	%t1 = insertelement <1 x i64> undef, i64 %t, i32 0
	%t2 = tail call <1 x i64> @llvm.x86.mmx.pslli.q(<1 x i64> %t1, i32 48)
	ret <1 x i64> %t2
}
