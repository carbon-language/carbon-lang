; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; These duplicate arm64 tests in vmax.ll

declare <8 x i8> @llvm.arm.neon.vpmaxs.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vpmaxu.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_smaxp_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: test_smaxp_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vpmaxs.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: smaxp v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_umaxp_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
  %tmp1 = call <8 x i8> @llvm.arm.neon.vpmaxu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: umaxp v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vpmaxs.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vpmaxu.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_smaxp_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_smaxp_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vpmaxs.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: smaxp v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_umaxp_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_umaxp_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vpmaxu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: umaxp v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_smaxp_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_smaxp_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vpmaxs.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: smaxp v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_umaxp_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_umaxp_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vpmaxu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: umaxp v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}


declare <8 x i16> @llvm.arm.neon.vpmaxs.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vpmaxu.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_smaxp_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_smaxp_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vpmaxs.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: smaxp v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_umaxp_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_umaxp_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vpmaxu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: umaxp v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}


declare <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_smaxp_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_smaxp_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vpmaxs.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: smaxp v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_umaxp_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_umaxp_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vpmaxu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: umaxp v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vpmaxs.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vpmaxu.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_smaxp_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_smaxp_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vpmaxs.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: smaxp v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_umaxp_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_umaxp_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vpmaxu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: umaxp v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

declare <8 x i8> @llvm.arm.neon.vpmins.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vpminu.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_sminp_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: test_sminp_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vpmins.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: sminp v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_uminp_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
  %tmp1 = call <8 x i8> @llvm.arm.neon.vpminu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: uminp v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vpmins.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vpminu.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_sminp_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_sminp_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vpmins.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: sminp v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_uminp_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uminp_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vpminu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: uminp v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_sminp_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_sminp_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vpmins.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sminp v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_uminp_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uminp_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vpminu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: uminp v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}


declare <8 x i16> @llvm.arm.neon.vpmins.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vpminu.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_sminp_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_sminp_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vpmins.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sminp v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_uminp_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uminp_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vpminu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: uminp v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}


declare <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_sminp_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_sminp_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vpmins.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sminp v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_uminp_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uminp_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vpminu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: uminp v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vpmins.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vpminu.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_sminp_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_sminp_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vpmins.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sminp v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_uminp_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uminp_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vpminu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: uminp v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

declare <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vpmaxs.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vpmaxs.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fmaxp_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fmaxp_v2f32:
        %val = call <2 x float> @llvm.arm.neon.vpmaxs.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fmaxp v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fmaxp_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fmaxp_v4f32:
        %val = call <4 x float> @llvm.arm.neon.vpmaxs.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fmaxp v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fmaxp_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fmaxp_v2f64:
        %val = call <2 x double> @llvm.arm.neon.vpmaxs.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fmaxp v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

declare <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vpmins.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vpmins.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fminp_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fminp_v2f32:
        %val = call <2 x float> @llvm.arm.neon.vpmins.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fminp v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fminp_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fminp_v4f32:
        %val = call <4 x float> @llvm.arm.neon.vpmins.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fminp v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fminp_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fminp_v2f64:
        %val = call <2 x double> @llvm.arm.neon.vpmins.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fminp v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

declare <2 x float> @llvm.aarch64.neon.vpmaxnm.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vpmaxnm.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vpmaxnm.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fmaxnmp_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fmaxnmp_v2f32:
        %val = call <2 x float> @llvm.aarch64.neon.vpmaxnm.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fmaxnmp v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fmaxnmp_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fmaxnmp_v4f32:
        %val = call <4 x float> @llvm.aarch64.neon.vpmaxnm.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fmaxnmp v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fmaxnmp_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fmaxnmp_v2f64:
        %val = call <2 x double> @llvm.aarch64.neon.vpmaxnm.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fmaxnmp v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

declare <2 x float> @llvm.aarch64.neon.vpminnm.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vpminnm.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vpminnm.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fminnmp_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fminnmp_v2f32:
        %val = call <2 x float> @llvm.aarch64.neon.vpminnm.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fminnmp v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fminnmp_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fminnmp_v4f32:
        %val = call <4 x float> @llvm.aarch64.neon.vpminnm.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fminnmp v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fminnmp_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fminnmp_v2f64:
        %val = call <2 x double> @llvm.aarch64.neon.vpminnm.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fminnmp v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

define i32 @test_vminv_s32(<2 x i32> %a) {
; CHECK-LABEL: test_vminv_s32
; CHECK: sminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %1 = tail call <1 x i32> @llvm.aarch64.neon.sminv.v1i32.v2i32(<2 x i32> %a)
  %2 = extractelement <1 x i32> %1, i32 0
  ret i32 %2
}

define i32 @test_vminv_u32(<2 x i32> %a) {
; CHECK-LABEL: test_vminv_u32
; CHECK: uminp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %1 = tail call <1 x i32> @llvm.aarch64.neon.uminv.v1i32.v2i32(<2 x i32> %a)
  %2 = extractelement <1 x i32> %1, i32 0
  ret i32 %2
}

define i32 @test_vmaxv_s32(<2 x i32> %a) {
; CHECK-LABEL: test_vmaxv_s32
; CHECK: smaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %1 = tail call <1 x i32> @llvm.aarch64.neon.smaxv.v1i32.v2i32(<2 x i32> %a)
  %2 = extractelement <1 x i32> %1, i32 0
  ret i32 %2
}

define i32 @test_vmaxv_u32(<2 x i32> %a) {
; CHECK-LABEL: test_vmaxv_u32
; CHECK: umaxp {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  %1 = tail call <1 x i32> @llvm.aarch64.neon.umaxv.v1i32.v2i32(<2 x i32> %a)
  %2 = extractelement <1 x i32> %1, i32 0
  ret i32 %2
}

declare <1 x i32> @llvm.aarch64.neon.uminv.v1i32.v2i32(<2 x i32>)
declare <1 x i32> @llvm.aarch64.neon.sminv.v1i32.v2i32(<2 x i32>)
declare <1 x i32> @llvm.aarch64.neon.umaxv.v1i32.v2i32(<2 x i32>)
declare <1 x i32> @llvm.aarch64.neon.smaxv.v1i32.v2i32(<2 x i32>)
