; RUN: llc < %s -mtriple=x86_64-pc-win32 -mcpu=core-avx2 -mattr=avx2,+fma3 | FileCheck %s

define <4 x float> @test_x86_fmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmadd231ps {{.*\(%r.*}}, %xmm
  %x = fmul <4 x float> %a0, %a1
  %res = fadd <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fmsub_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fmsub231ps {{.*\(%r.*}}, %xmm
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %x, %a2
  ret <4 x float> %res
}

define <4 x float> @test_x86_fnmadd_ps(<4 x float> %a0, <4 x float> %a1, <4 x float> %a2) {
  ; CHECK: fnmadd231ps {{.*\(%r.*}}, %xmm
  %x = fmul <4 x float> %a0, %a1
  %res = fsub <4 x float> %a2, %x
  ret <4 x float> %res
}

define <8 x float> @test_x86_fmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: vfmadd213ps	{{.*\(%r.*}}, %ymm
  %x = fmul <8 x float> %a0, %a1
  %res = fadd <8 x float> %x, %a2
  ret <8 x float> %res
}

define <4 x double> @test_x86_fmadd_pd_y(<4 x double> %a0, <4 x double> %a1, <4 x double> %a2) {
  ; CHECK: vfmadd231pd {{.*\(%r.*}}, %ymm
  %x = fmul <4 x double> %a0, %a1
  %res = fadd <4 x double> %x, %a2
  ret <4 x double> %res
}


define <8 x float> @test_x86_fmsub_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fmsub231ps {{.*\(%r.*}}, %ymm
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %x, %a2
  ret <8 x float> %res
}

define <8 x float> @test_x86_fnmadd_ps_y(<8 x float> %a0, <8 x float> %a1, <8 x float> %a2) {
  ; CHECK: fnmadd231ps {{.*\(%r.*}}, %ymm
  %x = fmul <8 x float> %a0, %a1
  %res = fsub <8 x float> %a2, %x
  ret <8 x float> %res
}

define float @test_x86_fnmadd_ss(float %a0, float %a1, float %a2) {
  ; CHECK: vfnmadd231ss    %xmm1, %xmm0, %xmm2
  %x = fmul float %a0, %a1
  %res = fsub float %a2, %x
  ret float %res
}

define double @test_x86_fnmadd_sd(double %a0, double %a1, double %a2) {
  ; CHECK: vfnmadd231sd    %xmm1, %xmm0, %xmm2
  %x = fmul double %a0, %a1
  %res = fsub double %a2, %x
  ret double %res
}

