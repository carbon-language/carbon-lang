; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

@var = global float 0.0

define void @foo() {
; CHECK-LABEL: foo:

; CHECK: stp d15, d14, [sp
; CHECK: stp d13, d12, [sp
; CHECK: stp d11, d10, [sp
; CHECK: stp d9, d8, [sp

  ; Create lots of live variables to exhaust the supply of
  ; caller-saved registers
  %val1 = load volatile float, float* @var
  %val2 = load volatile float, float* @var
  %val3 = load volatile float, float* @var
  %val4 = load volatile float, float* @var
  %val5 = load volatile float, float* @var
  %val6 = load volatile float, float* @var
  %val7 = load volatile float, float* @var
  %val8 = load volatile float, float* @var
  %val9 = load volatile float, float* @var
  %val10 = load volatile float, float* @var
  %val11 = load volatile float, float* @var
  %val12 = load volatile float, float* @var
  %val13 = load volatile float, float* @var
  %val14 = load volatile float, float* @var
  %val15 = load volatile float, float* @var
  %val16 = load volatile float, float* @var
  %val17 = load volatile float, float* @var
  %val18 = load volatile float, float* @var
  %val19 = load volatile float, float* @var
  %val20 = load volatile float, float* @var
  %val21 = load volatile float, float* @var
  %val22 = load volatile float, float* @var
  %val23 = load volatile float, float* @var
  %val24 = load volatile float, float* @var
  %val25 = load volatile float, float* @var
  %val26 = load volatile float, float* @var
  %val27 = load volatile float, float* @var
  %val28 = load volatile float, float* @var
  %val29 = load volatile float, float* @var
  %val30 = load volatile float, float* @var
  %val31 = load volatile float, float* @var
  %val32 = load volatile float, float* @var

  store volatile float %val1, float* @var
  store volatile float %val2, float* @var
  store volatile float %val3, float* @var
  store volatile float %val4, float* @var
  store volatile float %val5, float* @var
  store volatile float %val6, float* @var
  store volatile float %val7, float* @var
  store volatile float %val8, float* @var
  store volatile float %val9, float* @var
  store volatile float %val10, float* @var
  store volatile float %val11, float* @var
  store volatile float %val12, float* @var
  store volatile float %val13, float* @var
  store volatile float %val14, float* @var
  store volatile float %val15, float* @var
  store volatile float %val16, float* @var
  store volatile float %val17, float* @var
  store volatile float %val18, float* @var
  store volatile float %val19, float* @var
  store volatile float %val20, float* @var
  store volatile float %val21, float* @var
  store volatile float %val22, float* @var
  store volatile float %val23, float* @var
  store volatile float %val24, float* @var
  store volatile float %val25, float* @var
  store volatile float %val26, float* @var
  store volatile float %val27, float* @var
  store volatile float %val28, float* @var
  store volatile float %val29, float* @var
  store volatile float %val30, float* @var
  store volatile float %val31, float* @var
  store volatile float %val32, float* @var

; CHECK: ldp     d9, d8, [sp
; CHECK: ldp     d11, d10, [sp
; CHECK: ldp     d13, d12, [sp
; CHECK: ldp     d15, d14, [sp
  ret void
}
