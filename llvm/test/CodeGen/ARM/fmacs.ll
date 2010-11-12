; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NEON
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=A8

define float @t1(float %acc, float %a, float %b) {
entry:
; VFP2: t1:
; VFP2: vmla.f32

; NEON: t1:
; NEON: vmla.f32

; A8: t1:
; A8: vmul.f32
; A8: vadd.f32
	%0 = fmul float %a, %b
        %1 = fadd float %acc, %0
	ret float %1
}

define double @t2(double %acc, double %a, double %b) {
entry:
; VFP2: t2:
; VFP2: vmla.f64

; NEON: t2:
; NEON: vmla.f64

; A8: t2:
; A8: vmul.f64
; A8: vadd.f64
	%0 = fmul double %a, %b
        %1 = fadd double %acc, %0
	ret double %1
}

define float @t3(float %acc, float %a, float %b) {
entry:
; VFP2: t3:
; VFP2: vmla.f32

; NEON: t3:
; NEON: vmla.f32

; A8: t3:
; A8: vmul.f32
; A8: vadd.f32
	%0 = fmul float %a, %b
        %1 = fadd float %0, %acc
	ret float %1
}
