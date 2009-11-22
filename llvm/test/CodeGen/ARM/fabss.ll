; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=1 | FileCheck %s -check-prefix=NFP1
; RUN: llc < %s -march=arm -mattr=+neon -arm-use-neon-fp=0 | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
        %dum = fadd float %a, %b
	%0 = tail call float @fabsf(float %dum)
        %dum1 = fadd float %0, %b
	ret float %dum1
}

declare float @fabsf(float)

; VFP2: test:
; VFP2: 	vabs.f32	s1, s1

; NFP1: test:
; NFP1: 	vabs.f32	d1, d1
; NFP0: test:
; NFP0: 	vabs.f32	s1, s1

; CORTEXA8: test:
; CORTEXA8: 	vabs.f32	d1, d1
; CORTEXA9: test:
; CORTEXA9: 	vabs.f32	s1, s1
