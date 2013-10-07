; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define float @test_vrecpss_f32(float %a, float %b) {
; CHECK: test_vrecpss_f32
; CHECK: frecps {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = insertelement <1 x float> undef, float %a, i32 0
  %2 = insertelement <1 x float> undef, float %b, i32 0
  %3 = call <1 x float> @llvm.arm.neon.vrecps.v1f32(<1 x float> %1, <1 x float> %2)
  %4 = extractelement <1 x float> %3, i32 0
  ret float %4
}

define double @test_vrecpsd_f64(double %a, double %b) {
; CHECK: test_vrecpsd_f64
; CHECK: frecps {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %1 = insertelement <1 x double> undef, double %a, i32 0
  %2 = insertelement <1 x double> undef, double %b, i32 0
  %3 = call <1 x double> @llvm.arm.neon.vrecps.v1f64(<1 x double> %1, <1 x double> %2)
  %4 = extractelement <1 x double> %3, i32 0
  ret double %4
}

declare <1 x float> @llvm.arm.neon.vrecps.v1f32(<1 x float>, <1 x float>)
declare <1 x double> @llvm.arm.neon.vrecps.v1f64(<1 x double>, <1 x double>)

define float @test_vrsqrtss_f32(float %a, float %b) {
; CHECK: test_vrsqrtss_f32
; CHECK: frsqrts {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = insertelement <1 x float> undef, float %a, i32 0
  %2 = insertelement <1 x float> undef, float %b, i32 0
  %3 = call <1 x float> @llvm.arm.neon.vrsqrts.v1f32(<1 x float> %1, <1 x float> %2)
  %4 = extractelement <1 x float> %3, i32 0
  ret float %4
}

define double @test_vrsqrtsd_f64(double %a, double %b) {
; CHECK: test_vrsqrtsd_f64
; CHECK: frsqrts {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %1 = insertelement <1 x double> undef, double %a, i32 0
  %2 = insertelement <1 x double> undef, double %b, i32 0
  %3 = call <1 x double> @llvm.arm.neon.vrsqrts.v1f64(<1 x double> %1, <1 x double> %2)
  %4 = extractelement <1 x double> %3, i32 0
  ret double %4
}

declare <1 x float> @llvm.arm.neon.vrsqrts.v1f32(<1 x float>, <1 x float>)
declare <1 x double> @llvm.arm.neon.vrsqrts.v1f64(<1 x double>, <1 x double>)
