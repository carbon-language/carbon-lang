; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=1 | FileCheck %s -check-prefix=NFP1
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=0 | FileCheck %s -check-prefix=NFP0

define float @test(float %a, float %b) {
entry:
	%0 = fsub float %a, %b
	ret float %0
}

; VFP2: vsub.f32	s0, s1, s0
; NFP1: vsub.f32	d0, d1, d0
; NFP0: vsub.f32	s0, s1, s0
