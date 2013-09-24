; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64>)

define <1 x i64> @test_addp_v1i64(<2 x i64> %a) {
; CHECK: test_addp_v1i64:
        %val = call <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64> %a)
; CHECK: addp d0, v0.2d
        ret <1 x i64> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfadd(<2 x float>)

define <1 x float> @test_faddp_v1f32(<2 x float> %a) {
; CHECK: test_faddp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfadd(<2 x float> %a)
; CHECK: faddp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfaddq(<2 x double>)

define <1 x double> @test_faddp_v1f64(<2 x double> %a) {
; CHECK: test_faddp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfaddq(<2 x double> %a)
; CHECK: faddp d0, v0.2d
        ret <1 x double> %val
}


declare <1 x float> @llvm.aarch64.neon.vpmax(<2 x float>)

define <1 x float> @test_fmaxp_v1f32(<2 x float> %a) {
; CHECK: test_fmaxp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpmax(<2 x float> %a)
; CHECK: fmaxp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpmaxq(<2 x double>)

define <1 x double> @test_fmaxp_v1f64(<2 x double> %a) {
; CHECK: test_fmaxp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpmaxq(<2 x double> %a)
; CHECK: fmaxp d0, v0.2d
        ret <1 x double> %val
}


declare <1 x float> @llvm.aarch64.neon.vpmin(<2 x float>)

define <1 x float> @test_fminp_v1f32(<2 x float> %a) {
; CHECK: test_fminp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpmin(<2 x float> %a)
; CHECK: fminp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpminq(<2 x double>)

define <1 x double> @test_fminp_v1f64(<2 x double> %a) {
; CHECK: test_fminp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpminq(<2 x double> %a)
; CHECK: fminp d0, v0.2d
        ret <1 x double> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfmaxnm(<2 x float>)

define <1 x float> @test_fmaxnmp_v1f32(<2 x float> %a) {
; CHECK: test_fmaxnmp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfmaxnm(<2 x float> %a)
; CHECK: fmaxnmp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfmaxnmq(<2 x double>)

define <1 x double> @test_fmaxnmp_v1f64(<2 x double> %a) {
; CHECK: test_fmaxnmp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfmaxnmq(<2 x double> %a)
; CHECK: fmaxnmp d0, v0.2d
        ret <1 x double> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfminnm(<2 x float>)

define <1 x float> @test_fminnmp_v1f32(<2 x float> %a) {
; CHECK: test_fminnmp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfminnm(<2 x float> %a)
; CHECK: fminnmp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfminnmq(<2 x double>)

define <1 x double> @test_fminnmp_v1f64(<2 x double> %a) {
; CHECK: test_fminnmp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfminnmq(<2 x double> %a)
; CHECK: fminnmp d0, v0.2d
        ret <1 x double> %val
}

