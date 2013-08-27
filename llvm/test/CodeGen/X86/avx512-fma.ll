; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -fp-contract=fast | FileCheck %s

; CHECK-LABEL: test_x86_fmadd_ps_z
; CHECK: vfmadd213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fadd <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmsub_ps_z
; CHECK: vfmsub213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fmsub_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %x, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmadd_ps_z
; CHECK: vfnmadd213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fnmadd_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %res = fsub <16 x float> %a2, %x
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fnmsub_ps_z
; CHECK: vfnmsub213ps     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <16 x float> @test_x86_fnmsub_ps_z(<16 x float> %a0, <16 x float> %a1, <16 x float> %a2) {
  %x = fmul <16 x float> %a0, %a1
  %y = fsub <16 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, 
                          float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00,
						  float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, 
						  float -0.000000e+00>, %x
  %res = fsub <16 x float> %y, %a2
  ret <16 x float> %res
}

; CHECK-LABEL: test_x86_fmadd_pd_z
; CHECK: vfmadd213pd     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test_x86_fmadd_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fadd <8 x double> %x, %a2
  ret <8 x double> %res
}

; CHECK-LABEL: test_x86_fmsub_pd_z
; CHECK: vfmsub213pd     %zmm2, %zmm1, %zmm0
; CHECK: ret
define <8 x double> @test_x86_fmsub_pd_z(<8 x double> %a0, <8 x double> %a1, <8 x double> %a2) {
  %x = fmul <8 x double> %a0, %a1
  %res = fsub <8 x double> %x, %a2
  ret <8 x double> %res
}

define double @test_x86_fmsub_sd_z(double %a0, double %a1, double %a2) {
  %x = fmul double %a0, %a1
  %res = fsub double %x, %a2
  ret double %res
}

;CHECK-LABEL: test132_br
;CHECK: vfmadd132ps  LCP{{.*}}(%rip){1to16}
;CHECK: ret
define <16 x float> @test132_br(<16 x float> %a1, <16 x float> %a2) nounwind {
  %b1 = fmul <16 x float> %a1, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  %b2 = fadd <16 x float> %b1, %a2
  ret <16 x float> %b2
}

;CHECK-LABEL: test213_br
;CHECK: vfmadd213ps  LCP{{.*}}(%rip){1to16}
;CHECK: ret
define <16 x float> @test213_br(<16 x float> %a1, <16 x float> %a2) nounwind {
  %b1 = fmul <16 x float> %a1, %a2
  %b2 = fadd <16 x float> %b1, <float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000, float 0x3FB99999A0000000>
  ret <16 x float> %b2
}
