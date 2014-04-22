; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; Intrinsic wrangling. Duplicates various arm64 tests.

declare <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64>)

define <1 x i64> @test_addp_v1i64(<2 x i64> %a) {
; CHECK: test_addp_v1i64:
; CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64> %a)
  ret <1 x i64> %val
}

declare float @llvm.aarch64.neon.vpfadd.f32.v2f32(<2 x float>)

define float @test_faddp_f32(<2 x float> %a) {
; CHECK: test_faddp_f32:
; CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %val = call float @llvm.aarch64.neon.vpfadd.f32.v2f32(<2 x float> %a)
  ret float %val
}

declare double @llvm.aarch64.neon.vpfadd.f64.v2f64(<2 x double>)

define double @test_faddp_f64(<2 x double> %a) {
; CHECK: test_faddp_f64:
; CHECK: faddp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call double @llvm.aarch64.neon.vpfadd.f64.v2f64(<2 x double> %a)
  ret double %val
}


declare float @llvm.aarch64.neon.vpmax.f32.v2f32(<2 x float>)

define float @test_fmaxp_f32(<2 x float> %a) {
; CHECK: test_fmaxp_f32:
; CHECK: fmaxp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %val = call float @llvm.aarch64.neon.vpmax.f32.v2f32(<2 x float> %a)
  ret float %val
}

declare double @llvm.aarch64.neon.vpmax.f64.v2f64(<2 x double>)

define double @test_fmaxp_f64(<2 x double> %a) {
; CHECK: test_fmaxp_f64:
; CHECK: fmaxp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call double @llvm.aarch64.neon.vpmax.f64.v2f64(<2 x double> %a)
  ret double %val
}

declare float @llvm.aarch64.neon.vpmin.f32.v2f32(<2 x float>)

define float @test_fminp_f32(<2 x float> %a) {
; CHECK: test_fminp_f32:
; CHECK: fminp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %val = call float @llvm.aarch64.neon.vpmin.f32.v2f32(<2 x float> %a)
  ret float %val
}

declare double @llvm.aarch64.neon.vpmin.f64.v2f64(<2 x double>)

define double @test_fminp_f64(<2 x double> %a) {
; CHECK: test_fminp_f64:
; CHECK: fminp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call double @llvm.aarch64.neon.vpmin.f64.v2f64(<2 x double> %a)
  ret double %val
}

declare float @llvm.aarch64.neon.vpfmaxnm.f32.v2f32(<2 x float>)

define float @test_fmaxnmp_f32(<2 x float> %a) {
; CHECK: test_fmaxnmp_f32:
; CHECK: fmaxnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %val = call float @llvm.aarch64.neon.vpfmaxnm.f32.v2f32(<2 x float> %a)
  ret float %val
}

declare double @llvm.aarch64.neon.vpfmaxnm.f64.v2f64(<2 x double>)

define double @test_fmaxnmp_f64(<2 x double> %a) {
; CHECK: test_fmaxnmp_f64:
; CHECK: fmaxnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call double @llvm.aarch64.neon.vpfmaxnm.f64.v2f64(<2 x double> %a)
  ret double %val
}

declare float @llvm.aarch64.neon.vpfminnm.f32.v2f32(<2 x float>)

define float @test_fminnmp_f32(<2 x float> %a) {
; CHECK: test_fminnmp_f32:
; CHECK: fminnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %val = call float @llvm.aarch64.neon.vpfminnm.f32.v2f32(<2 x float> %a)
  ret float %val
}

declare double @llvm.aarch64.neon.vpfminnm.f64.v2f64(<2 x double>)

define double @test_fminnmp_f64(<2 x double> %a) {
; CHECK: test_fminnmp_f64:
; CHECK: fminnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %val = call double @llvm.aarch64.neon.vpfminnm.f64.v2f64(<2 x double> %a)
  ret double %val
}

define float @test_vaddv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vaddv_f32
; CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpfadd.f32.v2f32(<2 x float> %a)
  ret float %1
}

define float @test_vaddvq_f32(<4 x float> %a) {
; CHECK-LABEL: test_vaddvq_f32
; CHECK: faddp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpfadd.f32.v4f32(<4 x float> %a)
  ret float %1
}

define double @test_vaddvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vaddvq_f64
; CHECK: faddp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call double @llvm.aarch64.neon.vpfadd.f64.v2f64(<2 x double> %a)
  ret double %1
}

define float @test_vmaxv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vmaxv_f32
; CHECK: fmaxp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpmax.f32.v2f32(<2 x float> %a)
  ret float %1
}

define double @test_vmaxvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vmaxvq_f64
; CHECK: fmaxp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call double @llvm.aarch64.neon.vpmax.f64.v2f64(<2 x double> %a)
  ret double %1
}

define float @test_vminv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vminv_f32
; CHECK: fminp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpmin.f32.v2f32(<2 x float> %a)
  ret float %1
}

define double @test_vminvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vminvq_f64
; CHECK: fminp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call double @llvm.aarch64.neon.vpmin.f64.v2f64(<2 x double> %a)
  ret double %1
}

define double @test_vmaxnmvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vmaxnmvq_f64
; CHECK: fmaxnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call double @llvm.aarch64.neon.vpfmaxnm.f64.v2f64(<2 x double> %a)
  ret double %1
}

define float @test_vmaxnmv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vmaxnmv_f32
; CHECK: fmaxnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpfmaxnm.f32.v2f32(<2 x float> %a)
  ret float %1
}

define double @test_vminnmvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vminnmvq_f64
; CHECK: fminnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call double @llvm.aarch64.neon.vpfminnm.f64.v2f64(<2 x double> %a)
  ret double %1
}

define float @test_vminnmv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vminnmv_f32
; CHECK: fminnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = call float @llvm.aarch64.neon.vpfminnm.f32.v2f32(<2 x float> %a)
  ret float %1
}

define <2 x i64> @test_vpaddq_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpaddq_s64
; CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  %1 = call <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %1
}

define <2 x i64> @test_vpaddq_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpaddq_u64
; CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  %1 = call <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %1
}

define i64 @test_vaddvq_s64(<2 x i64> %a) {
; CHECK-LABEL: test_vaddvq_s64
; CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64> %a)
  %2 = extractelement <1 x i64> %1, i32 0
  ret i64 %2
}

define i64 @test_vaddvq_u64(<2 x i64> %a) {
; CHECK-LABEL: test_vaddvq_u64
; CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = call <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64> %a)
  %2 = extractelement <1 x i64> %1, i32 0
  ret i64 %2
}

declare <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64>)

declare <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64>, <2 x i64>)

declare float @llvm.aarch64.neon.vpfadd.f32.v4f32(<4 x float>)
