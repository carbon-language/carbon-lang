; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

@varf32 = global float 0.0
@varf64 = global double 0.0

define void @check_float() {
; CHECK-LABEL: check_float:

  %val = load float* @varf32
  %newval1 = fadd float %val, 8.5
  store volatile float %newval1, float* @varf32
; CHECK-DAG: fmov [[EIGHT5:s[0-9]+]], #8.5

  %newval2 = fadd float %val, 128.0
  store volatile float %newval2, float* @varf32
; CHECK-DAG: ldr [[HARD:s[0-9]+]], [{{x[0-9]+}}, {{#?}}:lo12:.LCPI0_0

; CHECK: ret
  ret void
}

define void @check_double() {
; CHECK-LABEL: check_double:

  %val = load double* @varf64
  %newval1 = fadd double %val, 8.5
  store volatile double %newval1, double* @varf64
; CHECK-DAG: fmov {{d[0-9]+}}, #8.5

  %newval2 = fadd double %val, 128.0
  store volatile double %newval2, double* @varf64
; CHECK-DAG: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:.LCPI1_0

; CHECK: ret
  ret void
}
