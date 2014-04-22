; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s
; arm64 has these two tests in vabs.ll

define float @test_vabds_f32(float %a, float %b) {
; CHECK-LABEL: test_vabds_f32
; CHECK: fabd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %0 = call float @llvm.aarch64.neon.vabd.f32(float %a, float %a)
  ret float %0
}

define double @test_vabdd_f64(double %a, double %b) {
; CHECK-LABEL: test_vabdd_f64
; CHECK: fabd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = call double @llvm.aarch64.neon.vabd.f64(double %a, double %b)
  ret double %0
}

declare double @llvm.aarch64.neon.vabd.f64(double, double)
declare float @llvm.aarch64.neon.vabd.f32(float, float)
