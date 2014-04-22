; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; duplicates arm64 tests in vsqrt.ll

define float @test_vrecpss_f32(float %a, float %b) {
; CHECK: test_vrecpss_f32
; CHECK: frecps {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = call float @llvm.aarch64.neon.vrecps.f32(float %a, float %b)
  ret float %1
}

define double @test_vrecpsd_f64(double %a, double %b) {
; CHECK: test_vrecpsd_f64
; CHECK: frecps {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %1 = call double @llvm.aarch64.neon.vrecps.f64(double %a, double %b)
  ret double %1
}

declare float @llvm.aarch64.neon.vrecps.f32(float, float)
declare double @llvm.aarch64.neon.vrecps.f64(double, double)

define float @test_vrsqrtss_f32(float %a, float %b) {
; CHECK: test_vrsqrtss_f32
; CHECK: frsqrts {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = call float @llvm.aarch64.neon.vrsqrts.f32(float %a, float %b)
  ret float %1
}

define double @test_vrsqrtsd_f64(double %a, double %b) {
; CHECK: test_vrsqrtsd_f64
; CHECK: frsqrts {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %1 = call double @llvm.aarch64.neon.vrsqrts.f64(double %a, double %b)
  ret double %1
}

declare float @llvm.aarch64.neon.vrsqrts.f32(float, float)
declare double @llvm.aarch64.neon.vrsqrts.f64(double, double)

define float @test_vrecpes_f32(float %a) {
; CHECK: test_vrecpes_f32
; CHECK: frecpe {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %0 = call float @llvm.aarch64.neon.vrecpe.f32(float %a)
  ret float %0
}

define double @test_vrecped_f64(double %a) {
; CHECK: test_vrecped_f64
; CHECK: frecpe {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = call double @llvm.aarch64.neon.vrecpe.f64(double %a)
  ret double %0
}

declare float @llvm.aarch64.neon.vrecpe.f32(float)
declare double @llvm.aarch64.neon.vrecpe.f64(double)

define float @test_vrecpxs_f32(float %a) {
; CHECK: test_vrecpxs_f32
; CHECK: frecpx {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %0 = call float @llvm.aarch64.neon.vrecpx.f32(float %a)
  ret float %0
}

define double @test_vrecpxd_f64(double %a) {
; CHECK: test_vrecpxd_f64
; CHECK: frecpx {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = call double @llvm.aarch64.neon.vrecpx.f64(double %a)
  ret double %0
}

declare float @llvm.aarch64.neon.vrecpx.f32(float)
declare double @llvm.aarch64.neon.vrecpx.f64(double)

define float @test_vrsqrtes_f32(float %a) {
; CHECK: test_vrsqrtes_f32
; CHECK: frsqrte {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %0 = call float @llvm.aarch64.neon.vrsqrte.f32(float %a)
  ret float %0
}

define double @test_vrsqrted_f64(double %a) {
; CHECK: test_vrsqrted_f64
; CHECK: frsqrte {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = call double @llvm.aarch64.neon.vrsqrte.f64(double %a)
  ret double %0
}

declare float @llvm.aarch64.neon.vrsqrte.f32(float)
declare double @llvm.aarch64.neon.vrsqrte.f64(double)
