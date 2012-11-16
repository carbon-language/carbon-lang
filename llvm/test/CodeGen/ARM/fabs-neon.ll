; RUN: llc < %s -mtriple=armv7-eabi -float-abi=hard -mcpu=cortex-a8 | FileCheck %s

; CHECK: test:
; CHECK:         vabs.f32        q0, q0
define <4 x float> @test(<4 x float> %a) {
  %foo = call <4 x float> @llvm.fabs.v4f32(<4 x float> %a)
  ret <4 x float> %foo
}
declare <4 x float> @llvm.fabs.v4f32(<4 x float> %a)

; CHECK: test2:
; CHECK:        vabs.f32        d0, d0
define <2 x float> @test2(<2 x float> %a) {
  %foo = call <2 x float> @llvm.fabs.v2f32(<2 x float> %a)
    ret <2 x float> %foo
}
declare <2 x float> @llvm.fabs.v2f32(<2 x float> %a)
