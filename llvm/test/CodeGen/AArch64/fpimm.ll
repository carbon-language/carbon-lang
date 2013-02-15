; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@varf32 = global float 0.0
@varf64 = global double 0.0

define void @check_float() {
; CHECK: check_float:

  %val = load float* @varf32
  %newval1 = fadd float %val, 8.5
  store volatile float %newval1, float* @varf32
; CHECK: fmov {{s[0-9]+}}, #8.5

  %newval2 = fadd float %val, 128.0
  store volatile float %newval2, float* @varf32
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, #:lo12:.LCPI0_0

  ret void
}

define void @check_double() {
; CHECK: check_double:

  %val = load double* @varf64
  %newval1 = fadd double %val, 8.5
  store volatile double %newval1, double* @varf64
; CHECK: fmov {{d[0-9]+}}, #8.5

  %newval2 = fadd double %val, 128.0
  store volatile double %newval2, double* @varf64
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, #:lo12:.LCPI1_0

  ret void
}
