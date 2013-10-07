; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define i16 @test_vqdmulhh_s16(i16 %a, i16 %b) {
; CHECK: test_vqdmulhh_s16
; CHECK: sqdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
  %1 = insertelement <1 x i16> undef, i16 %a, i32 0
  %2 = insertelement <1 x i16> undef, i16 %b, i32 0
  %3 = call <1 x i16> @llvm.arm.neon.vqdmulh.v1i16(<1 x i16> %1, <1 x i16> %2)
  %4 = extractelement <1 x i16> %3, i32 0
  ret i16 %4
}

define i32 @test_vqdmulhs_s32(i32 %a, i32 %b) {
; CHECK: test_vqdmulhs_s32
; CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = insertelement <1 x i32> undef, i32 %a, i32 0
  %2 = insertelement <1 x i32> undef, i32 %b, i32 0
  %3 = call <1 x i32> @llvm.arm.neon.vqdmulh.v1i32(<1 x i32> %1, <1 x i32> %2)
  %4 = extractelement <1 x i32> %3, i32 0
  ret i32 %4
}

declare <1 x i16> @llvm.arm.neon.vqdmulh.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i32> @llvm.arm.neon.vqdmulh.v1i32(<1 x i32>, <1 x i32>)

define i16 @test_vqrdmulhh_s16(i16 %a, i16 %b) {
; CHECK: test_vqrdmulhh_s16
; CHECK: sqrdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{h[0-9]+}}
  %1 = insertelement <1 x i16> undef, i16 %a, i32 0
  %2 = insertelement <1 x i16> undef, i16 %b, i32 0
  %3 = call <1 x i16> @llvm.arm.neon.vqrdmulh.v1i16(<1 x i16> %1, <1 x i16> %2)
  %4 = extractelement <1 x i16> %3, i32 0
  ret i16 %4
}

define i32 @test_vqrdmulhs_s32(i32 %a, i32 %b) {
; CHECK: test_vqrdmulhs_s32
; CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = insertelement <1 x i32> undef, i32 %a, i32 0
  %2 = insertelement <1 x i32> undef, i32 %b, i32 0
  %3 = call <1 x i32> @llvm.arm.neon.vqrdmulh.v1i32(<1 x i32> %1, <1 x i32> %2)
  %4 = extractelement <1 x i32> %3, i32 0
  ret i32 %4
}

declare <1 x i16> @llvm.arm.neon.vqrdmulh.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i32> @llvm.arm.neon.vqrdmulh.v1i32(<1 x i32>, <1 x i32>)

define float @test_vmulxs_f32(float %a, float %b) {
; CHECK: test_vmulxs_f32
; CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %1 = insertelement <1 x float> undef, float %a, i32 0
  %2 = insertelement <1 x float> undef, float %b, i32 0
  %3 = call <1 x float> @llvm.aarch64.neon.vmulx.v1f32(<1 x float> %1, <1 x float> %2)
  %4 = extractelement <1 x float> %3, i32 0
  ret float %4
}

define double @test_vmulxd_f64(double %a, double %b) {
; CHECK: test_vmulxd_f64
; CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %1 = insertelement <1 x double> undef, double %a, i32 0
  %2 = insertelement <1 x double> undef, double %b, i32 0
  %3 = call <1 x double> @llvm.aarch64.neon.vmulx.v1f64(<1 x double> %1, <1 x double> %2)
  %4 = extractelement <1 x double> %3, i32 0
  ret double %4
}

declare <1 x float> @llvm.aarch64.neon.vmulx.v1f32(<1 x float>, <1 x float>)
declare <1 x double> @llvm.aarch64.neon.vmulx.v1f64(<1 x double>, <1 x double>)
