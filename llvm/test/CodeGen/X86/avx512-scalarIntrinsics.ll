; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

declare <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float>, <4 x float>,<4 x float>, i8, i32)
define <4 x float>@test_int_x86_avx512_mask_scalef_ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 %x4) {
  ; CHECK-LABEL: test_int_x86_avx512_mask_scalef_ss:
  ; CHECK:    vscalefss %xmm1, %xmm0, %xmm2 {%k1}
  ; CHECK:    vscalefss {rn-sae}, %xmm1, %xmm0, %xmm0
    %res = call <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 %x4, i32 4)
    %res1 = call <4 x float> @llvm.x86.avx512.mask.scalef.ss(<4 x float> %x0, <4 x float> %x1, <4 x float> %x3, i8 -1, i32 8)
    %res2 = fadd <4 x float> %res, %res1
    ret <4 x float> %res2
}

declare <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double>, <2 x double>,<2 x double>, i8, i32)
define <2 x double>@test_int_x86_avx512_mask_scalef_sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 %x4) {
  ; CHECK-LABEL: test_int_x86_avx512_mask_scalef_sd:
  ; CHECK:    vscalefsd %xmm1, %xmm0, %xmm2 {%k1}
  ; CHECK:    vscalefsd {rn-sae}, %xmm1, %xmm0, %xmm0
    %res = call <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 %x4, i32 4)
    %res1 = call <2 x double> @llvm.x86.avx512.mask.scalef.sd(<2 x double> %x0, <2 x double> %x1, <2 x double> %x3, i8 -1, i32 8)
    %res2 = fadd <2 x double> %res, %res1
    ret <2 x double> %res2
}
