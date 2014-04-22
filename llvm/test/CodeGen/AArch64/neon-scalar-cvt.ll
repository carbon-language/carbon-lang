; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; arm64 has a different approach to scalars. Discarding.

define float @test_vcvts_f32_s32(i32 %a) {
; CHECK: test_vcvts_f32_s32
; CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtint2fps.f32.v1i32(<1 x i32> %vcvtf.i)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtint2fps.f32.v1i32(<1 x i32>)

define double @test_vcvtd_f64_s64(i64 %a) {
; CHECK: test_vcvtd_f64_s64
; CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtint2fps.f64.v1i64(<1 x i64> %vcvtf.i)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtint2fps.f64.v1i64(<1 x i64>)

define float @test_vcvts_f32_u32(i32 %a) {
; CHECK: test_vcvts_f32_u32
; CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtint2fpu.f32.v1i32(<1 x i32> %vcvtf.i)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtint2fpu.f32.v1i32(<1 x i32>)

define double @test_vcvtd_f64_u64(i64 %a) {
; CHECK: test_vcvtd_f64_u64
; CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vcvtf.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtint2fpu.f64.v1i64(<1 x i64> %vcvtf.i)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtint2fpu.f64.v1i64(<1 x i64>)

define float @test_vcvts_n_f32_s32(i32 %a) {
; CHECK: test_vcvts_n_f32_s32
; CHECK: scvtf {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtfxs2fp.n.f32.v1i32(<1 x i32> %vcvtf, i32 1)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtfxs2fp.n.f32.v1i32(<1 x i32>, i32)

define double @test_vcvtd_n_f64_s64(i64 %a) {
; CHECK: test_vcvtd_n_f64_s64
; CHECK: scvtf {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtfxs2fp.n.f64.v1i64(<1 x i64> %vcvtf, i32 1)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtfxs2fp.n.f64.v1i64(<1 x i64>, i32)

define float @test_vcvts_n_f32_u32(i32 %a) {
; CHECK: test_vcvts_n_f32_u32
; CHECK: ucvtf {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i32> undef, i32 %a, i32 0
  %0 = call float @llvm.aarch64.neon.vcvtfxu2fp.n.f32.v1i32(<1 x i32> %vcvtf, i32 1)
  ret float %0
}

declare float @llvm.aarch64.neon.vcvtfxu2fp.n.f32.v1i32(<1 x i32>, i32)

define double @test_vcvtd_n_f64_u64(i64 %a) {
; CHECK: test_vcvtd_n_f64_u64
; CHECK: ucvtf {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %vcvtf = insertelement <1 x i64> undef, i64 %a, i32 0
  %0 = call double @llvm.aarch64.neon.vcvtfxu2fp.n.f64.v1i64(<1 x i64> %vcvtf, i32 1)
  ret double %0
}

declare double @llvm.aarch64.neon.vcvtfxu2fp.n.f64.v1i64(<1 x i64>, i32)

define i32 @test_vcvts_n_s32_f32(float %a) {
; CHECK: test_vcvts_n_s32_f32
; CHECK: fcvtzs {{s[0-9]+}}, {{s[0-9]+}}, #1
entry:
  %fcvtzs1 = call <1 x i32> @llvm.aarch64.neon.vcvtfp2fxs.n.v1i32.f32(float %a, i32 1)
  %0 = extractelement <1 x i32> %fcvtzs1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vcvtfp2fxs.n.v1i32.f32(float, i32)

define i64 @test_vcvtd_n_s64_f64(double %a) {
; CHECK: test_vcvtd_n_s64_f64
; CHECK: fcvtzs {{d[0-9]+}}, {{d[0-9]+}}, #1
entry:
  %fcvtzs1 = call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxs.n.v1i64.f64(double %a, i32 1)
  %0 = extractelement <1 x i64> %fcvtzs1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vcvtfp2fxs.n.v1i64.f64(double, i32)

define i32 @test_vcvts_n_u32_f32(float %a) {
; CHECK: test_vcvts_n_u32_f32
; CHECK: fcvtzu {{s[0-9]+}}, {{s[0-9]+}}, #32
entry:
  %fcvtzu1 = call <1 x i32> @llvm.aarch64.neon.vcvtfp2fxu.n.v1i32.f32(float %a, i32 32)
  %0 = extractelement <1 x i32> %fcvtzu1, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.aarch64.neon.vcvtfp2fxu.n.v1i32.f32(float, i32)

define i64 @test_vcvtd_n_u64_f64(double %a) {
; CHECK: test_vcvtd_n_u64_f64
; CHECK: fcvtzu {{d[0-9]+}}, {{d[0-9]+}}, #64
entry:
  %fcvtzu1 = tail call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxu.n.v1i64.f64(double %a, i32 64)
  %0 = extractelement <1 x i64> %fcvtzu1, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vcvtfp2fxu.n.v1i64.f64(double, i32)
