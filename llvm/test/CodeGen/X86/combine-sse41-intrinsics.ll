; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s


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


define <2 x double> @test_x86_sse41_blendv_pd(<2 x double> %a0, <2 x double> %a1) {
  %1 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> zeroinitializer)
  ret <2 x double> %1
}
; CHECK-LABEL: test_x86_sse41_blendv_pd
; CHECK-NOT: blendvpd
; CHECK: ret


define <4 x float> @test_x86_sse41_blendv_ps(<4 x float> %a0, <4 x float> %a1) {
  %1 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> zeroinitializer)
  ret <4 x float> %1
}
; CHECK-LABEL: test_x86_sse41_blendv_ps
; CHECK-NOT: blendvps
; CHECK: ret


define <16 x i8> @test_x86_sse41_pblendv_b(<16 x i8> %a0, <16 x i8> %a1) {
  %1 = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> zeroinitializer)
  ret <16 x i8> %1
}
; CHECK-LABEL: test_x86_sse41_pblendv_b
; CHECK-NOT: pblendvb
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


define <2 x double> @test2_x86_sse41_blendv_pd(<2 x double> %a0, <2 x double> %a1) {
  %Mask = bitcast <2 x i64> <i64 -1, i64 -1> to <2 x double>
  %1 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a0, <2 x double> %a1, <2 x double> %Mask )
  ret <2 x double> %1
}
; CHECK-LABEL: test2_x86_sse41_blendv_pd
; CHECK-NOT: blendvpd
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret


define <4 x float> @test2_x86_sse41_blendv_ps(<4 x float> %a0, <4 x float> %a1) {
  %Mask = bitcast <2 x i64> <i64 -1, i64 -1> to <4 x float>
  %1 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a0, <4 x float> %a1, <4 x float> %Mask)
  ret <4 x float> %1
}
; CHECK-LABEL: test2_x86_sse41_blendv_ps
; CHECK-NOT: blendvps
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret


define <16 x i8> @test2_x86_sse41_pblendv_b(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %a2) {
  %Mask = bitcast <2 x i64> <i64 -1, i64 -1> to <16 x i8>
  %1 = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %a0, <16 x i8> %a1, <16 x i8> %Mask)
  ret <16 x i8> %1
}
; CHECK-LABEL: test2_x86_sse41_pblendv_b
; CHECK-NOT: pblendvb
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


define <2 x double> @test3_x86_sse41_blendv_pd(<2 x double> %a0, <2 x double> %a1) {
  %1 = call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %a0, <2 x double> %a0, <2 x double> %a1 )
  ret <2 x double> %1
}
; CHECK-LABEL: test3_x86_sse41_blendv_pd
; CHECK-NOT: blendvpd
; CHECK: ret


define <4 x float> @test3_x86_sse41_blendv_ps(<4 x float> %a0, <4 x float> %a1) {
  %1 = call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %a0, <4 x float> %a0, <4 x float> %a1)
  ret <4 x float> %1
}
; CHECK-LABEL: test3_x86_sse41_blendv_ps
; CHECK-NOT: blendvps
; CHECK: ret


define <16 x i8> @test3_x86_sse41_pblendv_b(<16 x i8> %a0, <16 x i8> %a1) {
  %1 = call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %a0, <16 x i8> %a0, <16 x i8> %a1)
  ret <16 x i8> %1
}
; CHECK-LABEL: test3_x86_sse41_pblendv_b
; CHECK-NOT: pblendvb
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
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>)
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>)
declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i32)
declare <8 x i16> @llvm.x86.sse41.phminposuw(<8 x i16>)

