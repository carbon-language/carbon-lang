; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=avx2,+fma -fp-contract=fast | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=bdver2 -mattr=-fma4 -fp-contract=fast | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=bdver1 -fp-contract=fast | FileCheck %s --check-prefix=CHECK_FMA4

; CHECK-LABEL: test_x86_fmadd_ps_y_wide
; CHECK: vfmadd213ps
; CHECK: vfmadd213ps
; CHECK: ret
; CHECK_FMA4-LABEL: test_x86_fmadd_ps_y_wide
; CHECK_FMA4: vfmaddps
; CHECK_FMA4: vfmaddps
; CHECK_FMA4: ret
define <16 x float> @test_x86_fmadd_ps_y_wide(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fadd <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmsub_ps_y_wide
; CHECK: vfmsub213ps
; CHECK: vfmsub213ps
; CHECK: ret
; CHECK_FMA4-LABEL: test_x86_fmsub_ps_y_wide
; CHECK_FMA4: vfmsubps
; CHECK_FMA4: vfmsubps
; CHECK_FMA4: ret
define <16 x float> @test_x86_fmsub_ps_y_wide(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmadd_ps_y_wide
; CHECK: vfnmadd213ps
; CHECK: vfnmadd213ps
; CHECK: ret
; CHECK_FMA4-LABEL: test_x86_fnmadd_ps_y_wide
; CHECK_FMA4: vfnmaddps
; CHECK_FMA4: vfnmaddps
; CHECK_FMA4: ret
define <16 x float> @test_x86_fnmadd_ps_y_wide(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %a2, %x
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmsub_ps_y_wide
; CHECK: vfnmsub213ps
; CHECK: vfnmsub213ps
; CHECK: ret
define <16 x float> @test_x86_fnmsub_ps_y_wide(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %y = fsub <16 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %x
  %res = fsub <16 x float> %y, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmadd_pd_y_wide
; CHECK: vfmadd213pd
; CHECK: vfmadd213pd
; CHECK: ret
; CHECK_FMA4-LABEL: test_x86_fmadd_pd_y_wide
; CHECK_FMA4: vfmaddpd
; CHECK_FMA4: vfmaddpd
; CHECK_FMA4: ret
define <8 x double> @test_x86_fmadd_pd_y_wide(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fadd <8 x double> %x, %a2
  ret <8 x double> %res
}

; CHECK-LABEL: test_x86_fmsub_pd_y_wide
; CHECK: vfmsub213pd
; CHECK: vfmsub213pd
; CHECK: ret
; CHECK_FMA4-LABEL: test_x86_fmsub_pd_y_wide
; CHECK_FMA4: vfmsubpd
; CHECK_FMA4: vfmsubpd
; CHECK_FMA4: ret
define <8 x double> @test_x86_fmsub_pd_y_wide(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fsub <8 x double> %x, %a2
  ret <8 x double> %res
}
