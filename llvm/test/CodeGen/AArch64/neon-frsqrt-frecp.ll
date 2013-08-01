; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon  | FileCheck %s

; Set of tests for when the intrinsic is used.

declare <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vrsqrts.v2f64(<2 x double>, <2 x double>)

define <2 x float> @frsqrts_from_intr_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frsqrts v0.2s, v0.2s, v1.2s
        %val = call <2 x float> @llvm.arm.neon.vrsqrts.v2f32(<2 x float> %lhs, <2 x float> %rhs)
        ret <2 x float> %val
}

define <4 x float> @frsqrts_from_intr_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frsqrts v0.4s, v0.4s, v1.4s
        %val = call <4 x float> @llvm.arm.neon.vrsqrts.v4f32(<4 x float> %lhs, <4 x float> %rhs)
        ret <4 x float> %val
}

define <2 x double> @frsqrts_from_intr_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frsqrts v0.2d, v0.2d, v1.2d
        %val = call <2 x double> @llvm.arm.neon.vrsqrts.v2f64(<2 x double> %lhs, <2 x double> %rhs)
        ret <2 x double> %val
}

declare <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.arm.neon.vrecps.v2f64(<2 x double>, <2 x double>)

define <2 x float> @frecps_from_intr_v2f32(<2 x float> %lhs, <2 x float> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frecps v0.2s, v0.2s, v1.2s
        %val = call <2 x float> @llvm.arm.neon.vrecps.v2f32(<2 x float> %lhs, <2 x float> %rhs)
        ret <2 x float> %val
}

define <4 x float> @frecps_from_intr_v4f32(<4 x float> %lhs, <4 x float> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frecps v0.4s, v0.4s, v1.4s
        %val = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %lhs, <4 x float> %rhs)
        ret <4 x float> %val
}

define <2 x double> @frecps_from_intr_v2f64(<2 x double> %lhs, <2 x double> %rhs) {
; Using registers other than v0, v1 are possible, but would be odd.
; CHECK: frecps v0.2d, v0.2d, v1.2d
        %val = call <2 x double> @llvm.arm.neon.vrecps.v2f64(<2 x double> %lhs, <2 x double> %rhs)
        ret <2 x double> %val
}

