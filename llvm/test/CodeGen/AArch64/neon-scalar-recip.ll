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

define float @test_vrecpes_f32(float %a) {
; CHECK: test_vrecpes_f32
; CHECK: frecpe {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vrecpe.i = insertelement <1 x float> undef, float %a, i32 0
  %vrecpe1.i = tail call <1 x float> @llvm.arm.neon.vrecpe.v1f32(<1 x float> %vrecpe.i)
  %0 = extractelement <1 x float> %vrecpe1.i, i32 0
  ret float %0
}

define double @test_vrecped_f64(double %a) {
; CHECK: test_vrecped_f64
; CHECK: frecpe {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vrecpe.i = insertelement <1 x double> undef, double %a, i32 0
  %vrecpe1.i = tail call <1 x double> @llvm.arm.neon.vrecpe.v1f64(<1 x double> %vrecpe.i)
  %0 = extractelement <1 x double> %vrecpe1.i, i32 0
  ret double %0
}

declare <1 x float> @llvm.arm.neon.vrecpe.v1f32(<1 x float>)
declare <1 x double> @llvm.arm.neon.vrecpe.v1f64(<1 x double>)

define float @test_vrecpxs_f32(float %a) {
; CHECK: test_vrecpxs_f32
; CHECK: frecpx {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vrecpx.i = insertelement <1 x float> undef, float %a, i32 0
  %vrecpx1.i = tail call <1 x float> @llvm.aarch64.neon.vrecpx.v1f32(<1 x float> %vrecpx.i)
  %0 = extractelement <1 x float> %vrecpx1.i, i32 0
  ret float %0
}

define double @test_vrecpxd_f64(double %a) {
; CHECK: test_vrecpxd_f64
; CHECK: frecpx {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vrecpx.i = insertelement <1 x double> undef, double %a, i32 0
  %vrecpx1.i = tail call <1 x double> @llvm.aarch64.neon.vrecpx.v1f64(<1 x double> %vrecpx.i)
  %0 = extractelement <1 x double> %vrecpx1.i, i32 0
  ret double %0
}

declare <1 x float> @llvm.aarch64.neon.vrecpx.v1f32(<1 x float>)
declare <1 x double> @llvm.aarch64.neon.vrecpx.v1f64(<1 x double>)

define float @test_vrsqrtes_f32(float %a) {
; CHECK: test_vrsqrtes_f32
; CHECK: frsqrte {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vrsqrte.i = insertelement <1 x float> undef, float %a, i32 0
  %vrsqrte1.i = tail call <1 x float> @llvm.arm.neon.vrsqrte.v1f32(<1 x float> %vrsqrte.i)
  %0 = extractelement <1 x float> %vrsqrte1.i, i32 0
  ret float %0
}

define double @test_vrsqrted_f64(double %a) {
; CHECK: test_vrsqrted_f64
; CHECK: frsqrte {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vrsqrte.i = insertelement <1 x double> undef, double %a, i32 0
  %vrsqrte1.i = tail call <1 x double> @llvm.arm.neon.vrsqrte.v1f64(<1 x double> %vrsqrte.i)
  %0 = extractelement <1 x double> %vrsqrte1.i, i32 0
  ret double %0
}

declare <1 x float> @llvm.arm.neon.vrsqrte.v1f32(<1 x float>)
declare <1 x double> @llvm.arm.neon.vrsqrte.v1f64(<1 x double>)
