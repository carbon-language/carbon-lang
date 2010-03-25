; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %acc, float %a, float %b) {
entry:
	%0 = fmul float %a, %b
        %1 = fadd float %acc, %0
	ret float %1
}

; VFP2: test:
; VFP2: 	vmla.f32	s2, s1, s0

; NFP1: test:
; NFP1: 	vmul.f32	d0, d1, d0
; NFP0: test:
; NFP0: 	vmla.f32	s2, s1, s0

; CORTEXA8: test:
; CORTEXA8: 	vmul.f32	d0, d1, d0
; CORTEXA9: test:
; CORTEXA9: 	vmla.f32	s2, s1, s0
