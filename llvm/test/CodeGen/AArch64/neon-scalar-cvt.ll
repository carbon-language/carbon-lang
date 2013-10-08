; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define float @test_vcvts_f32_s32(i32 %a) {
; CHECK: test_vcvts_f32_s32
; CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vcvtf1.i = tail call <1 x float> @llvm.aarch64.neon.vcvtf32.s32(<1 x i32> %vcvtf.i)
  %0 = extractelement <1 x float> %vcvtf1.i, i32 0
  ret float %0
}

declare <1 x float> @llvm.aarch64.neon.vcvtf32.s32(<1 x i32>)

define double @test_vcvtd_f64_s64(i64 %a) {
; CHECK: test_vcvtd_f64_s64
; CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcvtf1.i = tail call <1 x double> @llvm.aarch64.neon.vcvtf64.s64(<1 x i64> %vcvtf.i)
  %0 = extractelement <1 x double> %vcvtf1.i, i32 0
  ret double %0
}

declare <1 x double> @llvm.aarch64.neon.vcvtf64.s64(<1 x i64>)

define float @test_vcvts_f32_u32(i32 %a) {
; CHECK: test_vcvts_f32_u32
; CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vcvtf1.i = tail call <1 x float> @llvm.aarch64.neon.vcvtf32.u32(<1 x i32> %vcvtf.i)
  %0 = extractelement <1 x float> %vcvtf1.i, i32 0
  ret float %0
}

declare <1 x float> @llvm.aarch64.neon.vcvtf32.u32(<1 x i32>)

define double @test_vcvtd_f64_u64(i64 %a) {
; CHECK: test_vcvtd_f64_u64
; CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vcvtf1.i = tail call <1 x double> @llvm.aarch64.neon.vcvtf64.u64(<1 x i64> %vcvtf.i)
  %0 = extractelement <1 x double> %vcvtf1.i, i32 0
  ret double %0
}

declare <1 x double> @llvm.aarch64.neon.vcvtf64.u64(<1 x i64>)
