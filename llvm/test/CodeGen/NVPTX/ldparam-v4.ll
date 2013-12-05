; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

declare <4 x float> @bar()

define void @foo(<4 x float>* %ptr) {
; CHECK: ld.param.v4.f32
  %val = tail call <4 x float> @bar()
  store <4 x float> %val, <4 x float>* %ptr
  ret void
}
