; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <8 x i8> @llvm.arm.neon.vmaxs.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vmaxu.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_smax_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: test_smax_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vmaxs.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: smax v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_umax_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
  %tmp1 = call <8 x i8> @llvm.arm.neon.vmaxu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: umax v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_smax_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_smax_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vmaxs.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: smax v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_umax_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_umax_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vmaxu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: umax v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_smax_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_smax_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vmaxs.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: smax v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_umax_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_umax_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vmaxu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: umax v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}


declare <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_smax_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_smax_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vmaxs.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: smax v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_umax_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_umax_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vmaxu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: umax v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}


declare <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_smax_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_smax_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vmaxs.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: smax v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_umax_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_umax_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vmaxu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: umax v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_smax_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_smax_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vmaxs.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: smax v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_umax_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_umax_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vmaxu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: umax v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

declare <8 x i8> @llvm.arm.neon.vmins.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vminu.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_smin_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: test_smin_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vmins.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: smin v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_umin_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
  %tmp1 = call <8 x i8> @llvm.arm.neon.vminu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: umin v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_smin_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_smin_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vmins.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: smin v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_umin_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_umin_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vminu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: umin v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_smin_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_smin_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vmins.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: smin v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_umin_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_umin_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vminu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: umin v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}


declare <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_smin_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_smin_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vmins.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: smin v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_umin_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_umin_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vminu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: umin v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}


declare <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_smin_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_smin_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vmins.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: smin v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_umin_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_umin_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vminu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: umin v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_smin_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_smin_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vmins.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: smin v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_umin_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_umin_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vminu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: umin v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

declare <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vmaxs.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fmax_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fmax_v2f32:
        %val = call <2 x float> @llvm.arm.neon.vmaxs.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fmax v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fmax_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fmax_v4f32:
        %val = call <4 x float> @llvm.arm.neon.vmaxs.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fmax v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fmax_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fmax_v2f64:
        %val = call <2 x double> @llvm.arm.neon.vmaxs.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fmax v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

declare <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vmins.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fmin_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fmin_v2f32:
        %val = call <2 x float> @llvm.arm.neon.vmins.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fmin v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fmin_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fmin_v4f32:
        %val = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fmin v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fmin_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fmin_v2f64:
        %val = call <2 x double> @llvm.arm.neon.vmins.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fmin v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}


declare <2 x float> @llvm.aarch64.neon.vmaxnm.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vmaxnm.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vmaxnm.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fmaxnm_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fmaxnm_v2f32:
        %val = call <2 x float> @llvm.aarch64.neon.vmaxnm.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fmaxnm v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fmaxnm_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fmaxnm_v4f32:
        %val = call <4 x float> @llvm.aarch64.neon.vmaxnm.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fmaxnm v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fmaxnm_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fmaxnm_v2f64:
        %val = call <2 x double> @llvm.aarch64.neon.vmaxnm.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fmaxnm v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}

declare <2 x float> @llvm.aarch64.neon.vminnm.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.aarch64.neon.vminnm.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vminnm.v2f64(<2 x double>, <2 x double>)

define <2 x float> @test_fminnm_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; CHECK: test_fminnm_v2f32:
        %val = call <2 x float> @llvm.aarch64.neon.vminnm.v2f32(<2 x float> %lhs, <2 x float> %rhs)
; CHECK: fminnm v0.2s, v0.2s, v1.2s
        ret <2 x float> %val
}

define <4 x float> @test_fminnm_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; CHECK: test_fminnm_v4f32:
        %val = call <4 x float> @llvm.aarch64.neon.vminnm.v4f32(<4 x float> %lhs, <4 x float> %rhs)
; CHECK: fminnm v0.4s, v0.4s, v1.4s
        ret <4 x float> %val
}

define <2 x double> @test_fminnm_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; CHECK: test_fminnm_v2f64:
        %val = call <2 x double> @llvm.aarch64.neon.vminnm.v2f64(<2 x double> %lhs, <2 x double> %rhs)
; CHECK: fminnm v0.2d, v0.2d, v1.2d
        ret <2 x double> %val
}
