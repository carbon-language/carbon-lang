; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -mtriple=arm-eabi -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -mtriple=arm-eabi -mcpu=cortex-a8 --enable-unsafe-fp-math | FileCheck %s -check-prefix=CORTEXA8U
; RUN: llc < %s -mtriple=arm-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8U
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
	%0 = fadd float %a, %b
	ret float %0
}

; VFP2-LABEL: test:
; VFP2: 	vadd.f32	s

; NFP1-LABEL: test:
; NFP1: 	vadd.f32	d
; NFP0-LABEL: test:
; NFP0: 	vadd.f32	s

; CORTEXA8-LABEL: test:
; CORTEXA8: 	vadd.f32	s
; CORTEXA8U-LABEL: test:
; CORTEXA8U: 	vadd.f32	d
; CORTEXA9-LABEL: test:
; CORTEXA9: 	vadd.f32	s
