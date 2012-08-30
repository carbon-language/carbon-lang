; RUN: llc -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a9 -mattr=+neon,+neonfp -float-abi=hard < %s | FileCheck %s

define <2 x float> @test_vmovs_via_vext_lane0to0(float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane0to0:
  %vec = insertelement <2 x float> %in, float %arg, i32 0
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d0, #1
; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane0to1(float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane0to1:
  %vec = insertelement <2 x float> %in, float %arg, i32 1
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vext.32 d1, d1, d0, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane1to0(float, float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane1to0:
  %vec = insertelement <2 x float> %in, float %arg, i32 0
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vext.32 d1, d0, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}

define <2 x float> @test_vmovs_via_vext_lane1to1(float, float %arg, <2 x float> %in) {
; CHECK: test_vmovs_via_vext_lane1to1:
  %vec = insertelement <2 x float> %in, float %arg, i32 1
  %res = fadd <2 x float> %vec, %vec

; CHECK: vext.32 d1, d0, d1, #1
; CHECK: vext.32 d1, d1, d1, #1
; CHECK: vadd.f32 {{d[0-9]+}}, d1, d1

  ret <2 x float> %res
}


define float @test_vmovs_via_vdup(float, float %ret, float %lhs, float %rhs) {
; CHECK: test_vmovs_via_vdup:

  ; Do an operation (which will end up NEON because of +neonfp) to convince the
  ; execution-domain pass that NEON is a good thing to use.
  %res = fadd float %ret, %ret
  ;  It makes sense for LLVM to do the addition in d0 here, because it's going
  ;  to be returned. This means it will want a "vmov s0, s1":
; CHECK: vdup.32 d0, d0[1]

  ret float %res
}

