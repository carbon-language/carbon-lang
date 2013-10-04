; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <2 x float> @test_vfma_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) {
; CHECK: test_vfma_lane_f32:
; CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %shuffle = shufflevector <2 x float> %v, <2 x float> undef, <2 x i32> <i32 1, i32 1>
  %mul = fmul <2 x float> %shuffle, %b
  %add = fadd <2 x float> %mul, %a
  ret <2 x float> %add
}

