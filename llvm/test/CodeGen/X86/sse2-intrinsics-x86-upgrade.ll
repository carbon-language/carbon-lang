; RUN: llc < %s -mtriple=x86_64-apple-darwin -march=x86 -mcpu=pentium4 -mattr=sse2 | FileCheck %s

define <2 x i64> @test_x86_sse2_psll_dq_bs(<2 x i64> %a0) {
  ; CHECK: pslldq {{.*#+}} xmm0 = zero,zero,zero,zero,zero,zero,zero,xmm0[0,1,2,3,4,5,6,7,8]
  %res = call <2 x i64> @llvm.x86.sse2.psll.dq.bs(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psll.dq.bs(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psrl_dq_bs(<2 x i64> %a0) {
  ; CHECK: psrldq {{.*#+}} xmm0 = xmm0[7,8,9,10,11,12,13,14,15],zero,zero,zero,zero,zero,zero,zero
  %res = call <2 x i64> @llvm.x86.sse2.psrl.dq.bs(<2 x i64> %a0, i32 7) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrl.dq.bs(<2 x i64>, i32) nounwind readnone

define <2 x i64> @test_x86_sse2_psll_dq(<2 x i64> %a0) {
  ; CHECK: pslldq {{.*#+}} xmm0 = zero,xmm0[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
  %res = call <2 x i64> @llvm.x86.sse2.psll.dq(<2 x i64> %a0, i32 8) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psll.dq(<2 x i64>, i32) nounwind readnone


define <2 x i64> @test_x86_sse2_psrl_dq(<2 x i64> %a0) {
  ; CHECK: psrldq {{.*#+}} xmm0 = xmm0[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],zero
  %res = call <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64> %a0, i32 8) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %res
}
declare <2 x i64> @llvm.x86.sse2.psrl.dq(<2 x i64>, i32) nounwind readnone
