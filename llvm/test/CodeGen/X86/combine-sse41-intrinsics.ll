; RUN: llc < %s -mtriple=x86_64-unknown -mcpu=corei7 | FileCheck %s


define <2 x double> @test_x86_sse41_blend_pd(<2 x double> %a0, <2 x double> %a1) {
  %1 = call <2 x double> @llvm.x86.sse41.blendpd(<2 x double> %a0, <2 x double> %a1, i32 0)
  ret <2 x double> %1
}
; CHECK-LABEL: test_x86_sse41_blend_pd
; CHECK-NOT: blendpd
; CHECK: ret


define <4 x float> @test_x86_sse41_blend_ps(<4 x float> %a0, <4 x float> %a1) {
  %1 = call <4 x float> @llvm.x86.sse41.blendps(<4 x float> %a0, <4 x float> %a1, i32 0)
  ret <4 x float> %1
}
; CHECK-LABEL: test_x86_sse41_blend_ps
; CHECK-NOT: blendps
; CHECK: ret


define <8 x i16> @test_x86_sse41_pblend_w(<8 x i16> %a0, <8 x i16> %a1) {
  %1 = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %a0, <8 x i16> %a1, i32 0)
  ret <8 x i16> %1
}
; CHECK-LABEL: test_x86_sse41_pblend_w
; CHECK-NOT: pblendw
; CHECK: ret


define <2 x double> @test2_x86_sse41_blend_pd(<2 x double> %a0, <2 x double> %a1) {
  %1 = call <2 x double> @llvm.x86.sse41.blendpd(<2 x double> %a0, <2 x double> %a1, i32 -1)
  ret <2 x double> %1
}
; CHECK-LABEL: test2_x86_sse41_blend_pd
; CHECK-NOT: blendpd
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret


define <4 x float> @test2_x86_sse41_blend_ps(<4 x float> %a0, <4 x float> %a1) {
  %1 = call <4 x float> @llvm.x86.sse41.blendps(<4 x float> %a0, <4 x float> %a1, i32 -1)
  ret <4 x float> %1
}
; CHECK-LABEL: test2_x86_sse41_blend_ps
; CHECK-NOT: blendps
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret


define <8 x i16> @test2_x86_sse41_pblend_w(<8 x i16> %a0, <8 x i16> %a1) {
  %1 = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %a0, <8 x i16> %a1, i32 -1)
  ret <8 x i16> %1
}
; CHECK-LABEL: test2_x86_sse41_pblend_w
; CHECK-NOT: pblendw
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret


define <2 x double> @test3_x86_sse41_blend_pd(<2 x double> %a0) {
  %1 = call <2 x double> @llvm.x86.sse41.blendpd(<2 x double> %a0, <2 x double> %a0, i32 7)
  ret <2 x double> %1
}
; CHECK-LABEL: test3_x86_sse41_blend_pd
; CHECK-NOT: blendpd
; CHECK: ret


define <4 x float> @test3_x86_sse41_blend_ps(<4 x float> %a0) {
  %1 = call <4 x float> @llvm.x86.sse41.blendps(<4 x float> %a0, <4 x float> %a0, i32 7)
  ret <4 x float> %1
}
; CHECK-LABEL: test3_x86_sse41_blend_ps
; CHECK-NOT: blendps
; CHECK: ret


define <8 x i16> @test3_x86_sse41_pblend_w(<8 x i16> %a0) {
  %1 = call <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16> %a0, <8 x i16> %a0, i32 7)
  ret <8 x i16> %1
}
; CHECK-LABEL: test3_x86_sse41_pblend_w
; CHECK-NOT: pblendw
; CHECK: ret


declare <2 x double> @llvm.x86.sse41.blendpd(<2 x double>, <2 x double>, i32)
declare <4 x float> @llvm.x86.sse41.blendps(<4 x float>, <4 x float>, i32)
declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i32)

