; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <2 x i32> @llvm.arm.neon.vacge.v2i32.v2f32(<2 x float>, <2 x float>)
declare <4 x i32> @llvm.arm.neon.vacge.v4i32.v4f32(<4 x float>, <4 x float>)
declare <2 x i64> @llvm.arm.neon.vacge.v2i64.v2f64(<2 x double>, <2 x double>)

define <2 x i32> @facge_from_intr_v2i32(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facge_from_intr_v2i32:
  %val = call <2 x i32> @llvm.arm.neon.vacge.v2i32.v2f32(<2 x float> %A, <2 x float> %B)
; CHECK: facge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  ret <2 x i32> %val
}
define <4 x i32> @facge_from_intr_v4i32( <4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facge_from_intr_v4i32:
  %val = call <4 x i32> @llvm.arm.neon.vacge.v4i32.v4f32(<4 x float> %A, <4 x float> %B)
; CHECK: facge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  ret <4 x i32> %val
}

define <2 x i64> @facge_from_intr_v2i64(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facge_from_intr_v2i64:
  %val = call <2 x i64> @llvm.arm.neon.vacge.v2i64.v2f64(<2 x double> %A, <2 x double> %B)
; CHECK: facge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  ret <2 x i64> %val
}

declare <2 x i32> @llvm.arm.neon.vacgt.v2i32.v2f32(<2 x float>, <2 x float>)
declare <4 x i32> @llvm.arm.neon.vacgt.v4i32.v4f32(<4 x float>, <4 x float>)
declare <2 x i64> @llvm.arm.neon.vacgt.v2i64.v2f64(<2 x double>, <2 x double>)

define <2 x i32> @facgt_from_intr_v2i32(<2 x float> %A, <2 x float> %B, <2 x float> %C) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facgt_from_intr_v2i32:
  %val = call <2 x i32> @llvm.arm.neon.vacgt.v2i32.v2f32(<2 x float> %A, <2 x float> %B)
; CHECK: facgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  ret <2 x i32> %val
}
define <4 x i32> @facgt_from_intr_v4i32( <4 x float> %A, <4 x float> %B) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facgt_from_intr_v4i32:
  %val = call <4 x i32> @llvm.arm.neon.vacgt.v4i32.v4f32(<4 x float> %A, <4 x float> %B)
; CHECK: facgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  ret <4 x i32> %val
}

define <2 x i64> @facgt_from_intr_v2i64(<2 x double> %A, <2 x double> %B) {
; Using registers other than v0, v1 and v2 are possible, but would be odd.
; CHECK: facgt_from_intr_v2i64:
  %val = call <2 x i64> @llvm.arm.neon.vacgt.v2i64.v2f64(<2 x double> %A, <2 x double> %B)
; CHECK: facgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  ret <2 x i64> %val
}

