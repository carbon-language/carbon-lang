; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma -fp-contract=fast | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma4,+fma -fp-contract=fast | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+fma4 -fp-contract=fast | FileCheck %s --check-prefix=CHECK_FMA4

define <4 x float> @test_x86_fmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fmadd_ps:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fadd <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fmsub_ps:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fnmadd_ps:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmadd213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %a2, %x
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fnmsub_ps:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmsub213ps %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ps:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x float> %a0, %a1
  %y = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %x
  %res = fsub <4 x float> %y, %a2
  ret <4 x float> %res
}

define <8 x float> @test_x86_fmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK-LABEL: test_x86_fmadd_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fadd <8 x float> %x, %a2
  ret <8 x float> %res
}

define <8 x float> @test_x86_fmsub_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK-LABEL: test_x86_fmsub_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %x, %a2
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK-LABEL: test_x86_fnmadd_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmadd213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %a2, %x
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmsub_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
; CHECK-LABEL: test_x86_fnmsub_ps_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmsub213ps %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ps_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubps %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <8 x float> %a0, %a1
  %y = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %x
  %res = fsub <8 x float> %y, %a2
  ret <8 x float> %res
}

define <4 x double> @test_x86_fmadd_pd_y(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
; CHECK-LABEL: test_x86_fmadd_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmadd213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_pd_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x double> %a0, %a1
  %res = fadd <4 x double> %x, %a2
  ret <4 x double> %res
}

define <4 x double> @test_x86_fmsub_pd_y(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
; CHECK-LABEL: test_x86_fmsub_pd_y:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmsub213pd %ymm2, %ymm1, %ymm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_pd_y:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubpd %ymm2, %ymm1, %ymm0, %ymm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <4 x double> %a0, %a1
  %res = fsub <4 x double> %x, %a2
  ret <4 x double> %res
}

define <2 x double> @test_x86_fmsub_pd(<2 x double> %a0, <2 x double> %a1, <2 x double> %a2) {
; CHECK-LABEL: test_x86_fmsub_pd:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmsub213pd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_pd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubpd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul <2 x double> %a0, %a1
  %res = fsub <2 x double> %x, %a2
  ret <2 x double> %res
}

define float @test_x86_fnmadd_ss(float %a0, float %a1, float %a2) {
; CHECK-LABEL: test_x86_fnmadd_ss:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmadd213ss %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_ss:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddss %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul float %a0, %a1
  %res = fsub float %a2, %x
  ret float %res
}

define double @test_x86_fnmadd_sd(double %a0, double %a1, double %a2) {
; CHECK-LABEL: test_x86_fnmadd_sd:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmadd213sd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmadd_sd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmaddsd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul double %a0, %a1
  %res = fsub double %a2, %x
  ret double %res
}

define double @test_x86_fmsub_sd(double %a0, double %a1, double %a2) {
; CHECK-LABEL: test_x86_fmsub_sd:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfmsub213sd %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_sd:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubsd %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fmul double %a0, %a1
  %res = fsub double %x, %a2
  ret double %res
}

define float @test_x86_fnmsub_ss(float %a0, float %a1, float %a2) {
; CHECK-LABEL: test_x86_fnmsub_ss:
; CHECK:       # BB#0:
; CHECK-NEXT:    vfnmsub213ss %xmm2, %xmm1, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fnmsub_ss:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfnmsubss %xmm2, %xmm1, %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = fsub float -0.000000e+00, %a0
  %y = fmul float %x, %a1
  %res = fsub float %y, %a2
  ret float %res
}

define <4 x float> @test_x86_fmadd_ps_load(<4 x float>* %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fmadd_ps_load:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmovaps (%rdi), %xmm2
; CHECK-NEXT:    vfmadd213ps %xmm1, %xmm2, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmadd_ps_load:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmaddps %xmm1, (%rdi), %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = load <4 x float>, <4 x float>* %a0
  %y = fmul <4 x float> %x, %a1
  %res = fadd <4 x float> %y, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_ps_load(<4 x float>* %a0, <4 x float> %a1, <4 x float> %a2) {
; CHECK-LABEL: test_x86_fmsub_ps_load:
; CHECK:       # BB#0:
; CHECK-NEXT:    vmovaps (%rdi), %xmm2
; CHECK-NEXT:    vfmsub213ps %xmm1, %xmm2, %xmm0
; CHECK-NEXT:    retq
;
; CHECK_FMA4-LABEL: test_x86_fmsub_ps_load:
; CHECK_FMA4:       # BB#0:
; CHECK_FMA4-NEXT:    vfmsubps %xmm1, (%rdi), %xmm0, %xmm0
; CHECK_FMA4-NEXT:    retq
  %x = load <4 x float>, <4 x float>* %a0
  %y = fmul <4 x float> %x, %a1
  %res = fsub <4 x float> %y, %a2
  ret <4 x float> %res
}

