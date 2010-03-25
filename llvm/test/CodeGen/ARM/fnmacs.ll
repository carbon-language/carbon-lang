; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NEON
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=NEONFP

define float @test(float %acc, float %a, float %b) {
entry:
; VFP2: vmls.f32
; NEON: vmls.f32

; NEONFP-NOT: vmls
; NEONFP-NOT: vmov.f32
; NEONFP:     vmul.f32
; NEONFP:     vsub.f32
; NEONFP:     vmov

	%0 = fmul float %a, %b
        %1 = fsub float %acc, %0
	ret float %1
}

