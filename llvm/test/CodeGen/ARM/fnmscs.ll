; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NEON
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=A8

define float @t1(float %acc, float %a, float %b) nounwind {
entry:
; VFP2: t1:
; VFP2: vnmla.f32

; NEON: t1:
; NEON: vnmla.f32

; A8: t1:
; A8: vnmul.f32 s0, s1, s0
; A8: vsub.f32 d0, d0, d1
	%0 = fmul float %a, %b
	%1 = fsub float -0.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}

define float @t2(float %acc, float %a, float %b) nounwind {
entry:
; VFP2: t2:
; VFP2: vnmla.f32

; NEON: t2:
; NEON: vnmla.f32

; A8: t2:
; A8: vnmul.f32 s0, s1, s0
; A8: vsub.f32 d0, d0, d1
	%0 = fmul float %a, %b
	%1 = fmul float -1.0, %0
        %2 = fsub float %1, %acc
	ret float %2
}

define double @t3(double %acc, double %a, double %b) nounwind {
entry:
; VFP2: t3:
; VFP2: vnmla.f64

; NEON: t3:
; NEON: vnmla.f64

; A8: t3:
; A8: vnmul.f64 d16, d16, d17
; A8: vsub.f64 d16, d16, d17
	%0 = fmul double %a, %b
	%1 = fsub double -0.0, %0
        %2 = fsub double %1, %acc
	ret double %2
}

define double @t4(double %acc, double %a, double %b) nounwind {
entry:
; VFP2: t4:
; VFP2: vnmla.f64

; NEON: t4:
; NEON: vnmla.f64

; A8: t4:
; A8: vnmul.f64 d16, d16, d17
; A8: vsub.f64 d16, d16, d17
	%0 = fmul double %a, %b
	%1 = fmul double -1.0, %0
        %2 = fsub double %1, %acc
	ret double %2
}
