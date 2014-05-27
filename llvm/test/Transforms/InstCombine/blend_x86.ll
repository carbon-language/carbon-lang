; RUN: opt < %s -instcombine -mtriple=x86_64-apple-macosx -mcpu=core-avx2 -S | FileCheck %s

define <2 x double> @constant_blendvpd(<2 x double> %xy, <2 x double> %ab) {
; CHECK-LABEL: @constant_blendvpd
; CHECK: select <2 x i1> <i1 true, i1 false>, <2 x double> %ab, <2 x double> %xy
  %1 = tail call <2 x double> @llvm.x86.sse41.blendvpd(<2 x double> %xy, <2 x double> %ab, <2 x double> <double 0xFFFFFFFFE0000000, double 0.000000e+00>)
  ret <2 x double> %1
}

define <4 x float> @constant_blendvps(<4 x float> %xyzw, <4 x float> %abcd) {
; CHECK-LABEL: @constant_blendvps
; CHECK: select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x float> %abcd, <4 x float> %xyzw
  %1 = tail call <4 x float> @llvm.x86.sse41.blendvps(<4 x float> %xyzw, <4 x float> %abcd, <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xFFFFFFFFE0000000>)
  ret <4 x float> %1
}

define <16 x i8> @constant_pblendvb(<16 x i8> %xyzw, <16 x i8> %abcd) {
; CHECK-LABEL: @constant_pblendvb
; CHECK: select <16 x i1> <i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false>, <16 x i8> %abcd, <16 x i8> %xyzw
  %1 = tail call <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8> %xyzw, <16 x i8> %abcd, <16 x i8> <i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0, i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0>)
  ret <16 x i8> %1
}

define <4 x double> @constant_blendvpd_avx(<4 x double> %xy, <4 x double> %ab) {
; CHECK-LABEL: @constant_blendvpd_avx
; CHECK: select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %ab, <4 x double> %xy
  %1 = tail call <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double> %xy, <4 x double> %ab, <4 x double> <double 0xFFFFFFFFE0000000, double 0.000000e+00, double 0xFFFFFFFFE0000000, double 0.000000e+00>)
  ret <4 x double> %1
}

define <8 x float> @constant_blendvps_avx(<8 x float> %xyzw, <8 x float> %abcd) {
; CHECK-LABEL: @constant_blendvps_avx
; CHECK: select <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true>,  <8 x float> %abcd, <8 x float> %xyzw
  %1 = tail call <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float> %xyzw, <8 x float> %abcd, <8 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xFFFFFFFFE0000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0xFFFFFFFFE0000000>)
  ret <8 x float> %1
}

define <32 x i8> @constant_pblendvb_avx2(<32 x i8> %xyzw, <32 x i8> %abcd) {
; CHECK-LABEL: @constant_pblendvb_avx2
; CHECK: select <32 x i1> <i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false>, <32 x i8> %abcd, <32 x i8> %xyzw
  %1 = tail call <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8> %xyzw, <32 x i8> %abcd,
        <32 x i8> <i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0,
                   i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0,
                   i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0,
                   i8 0, i8 0, i8 255, i8 0, i8 255, i8 255, i8 255, i8 0>)
  ret <32 x i8> %1
}

declare <16 x i8> @llvm.x86.sse41.pblendvb(<16 x i8>, <16 x i8>, <16 x i8>)
declare <4 x float> @llvm.x86.sse41.blendvps(<4 x float>, <4 x float>, <4 x float>)
declare <2 x double> @llvm.x86.sse41.blendvpd(<2 x double>, <2 x double>, <2 x double>)

declare <32 x i8> @llvm.x86.avx2.pblendvb(<32 x i8>, <32 x i8>, <32 x i8>)
declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>)
