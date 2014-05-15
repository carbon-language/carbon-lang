; RUN: llc < %s -march=x86-64 -mcpu=corei7-avx | FileCheck %s


define <4 x double> @test_x86_avx_blend_pd_256(<4 x double> %a0) {
  %1 = call <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double> %a0, <4 x double> %a0, i32 7)
  ret <4 x double> %1
}
; CHECK-LABEL: test_x86_avx_blend_pd_256
; CHECK-NOT: vblendpd
; CHECK: ret


define <8 x float> @test_x86_avx_blend_ps_256(<8 x float> %a0) {
  %1 = call <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float> %a0, <8 x float> %a0, i32 7)
  ret <8 x float> %1
}
; CHECK-LABEL: test_x86_avx_blend_ps_256
; CHECK-NOT: vblendps
; CHECK: ret


define <4 x double> @test_x86_avx_blendv_pd_256(<4 x double> %a0, <4 x double> %a1) {
  %1 = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a0, <4 x double> %a0, <4 x double> %a1)
  ret <4 x double> %1
}
; CHECK-LABEL: test_x86_avx_blendv_pd_256
; CHECK-NOT: vblendvpd
; CHECK: ret


define <8 x float> @test_x86_avx_blendv_ps_256(<8 x float> %a0, <8 x float> %a1) {
  %1 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a0, <8 x float> %a0, <8 x float> %a1)
  ret <8 x float> %1
}
; CHECK-LABEL: test_x86_avx_blendv_ps_256
; CHECK-NOT: vblendvps
; CHECK: ret


define <4 x double> @test2_x86_avx_blend_pd_256(<4 x double> %a0, <4 x double> %a1) {
  %1 = call <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double> %a0, <4 x double> %a1, i32 0)
  ret <4 x double> %1
}
; CHECK-LABEL: test2_x86_avx_blend_pd_256
; CHECK-NOT: vblendpd
; CHECK: ret


define <8 x float> @test2_x86_avx_blend_ps_256(<8 x float> %a0, <8 x float> %a1) {
  %1 = call <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float> %a0, <8 x float> %a1, i32 0)
  ret <8 x float> %1
}
; CHECK-LABEL: test2_x86_avx_blend_ps_256
; CHECK-NOT: vblendps
; CHECK: ret


define <4 x double> @test2_x86_avx_blendv_pd_256(<4 x double> %a0, <4 x double> %a1) {
  %1 = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> zeroinitializer)
  ret <4 x double> %1
}
; CHECK-LABEL: test2_x86_avx_blendv_pd_256
; CHECK-NOT: vblendvpd
; CHECK: ret


define <8 x float> @test2_x86_avx_blendv_ps_256(<8 x float> %a0, <8 x float> %a1) {
  %1 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> zeroinitializer)
  ret <8 x float> %1
}
; CHECK-LABEL: test2_x86_avx_blendv_ps_256
; CHECK-NOT: vblendvps
; CHECK: ret


define <4 x double> @test3_x86_avx_blend_pd_256(<4 x double> %a0, <4 x double> %a1) {
  %1 = call <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double> %a0, <4 x double> %a1, i32 -1)
  ret <4 x double> %1
}
; CHECK-LABEL: test3_x86_avx_blend_pd_256
; CHECK-NOT: vblendpd
; CHECK: ret


define <8 x float> @test3_x86_avx_blend_ps_256(<8 x float> %a0, <8 x float> %a1) {
  %1 = call <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float> %a0, <8 x float> %a1, i32 -1)
  ret <8 x float> %1
}
; CHECK-LABEL: test3_x86_avx_blend_ps_256
; CHECK-NOT: vblendps
; CHECK: ret


define <4 x double> @test3_x86_avx_blendv_pd_256(<4 x double> %a0, <4 x double> %a1) {
  %Mask = bitcast <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1> to <4 x double>
  %1 = call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %a0, <4 x double> %a1, <4 x double> %Mask)
  ret <4 x double> %1
}
; CHECK-LABEL: test3_x86_avx_blendv_pd_256
; CHECK-NOT: vblendvpd
; CHECK: ret


define <8 x float> @test3_x86_avx_blendv_ps_256(<8 x float> %a0, <8 x float> %a1) {
  %Mask = bitcast <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1> to <8 x float>
  %1 = call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %a0, <8 x float> %a1, <8 x float> %Mask)
  ret <8 x float> %1
}
; CHECK-LABEL: test3_x86_avx_blendv_ps_256
; CHECK-NOT: vblendvps
; CHECK: ret



declare <4 x double> @llvm.x86.avx.blend.pd.256(<4 x double>, <4 x double>, i32)
declare <8 x float> @llvm.x86.avx.blend.ps.256(<8 x float>, <8 x float>, i32)
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>)
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>)

