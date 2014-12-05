; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,-slow-unaligned-mem-32 | FileCheck %s --check-prefix=ALL --check-prefix=FAST32
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+slow-unaligned-mem-32 | FileCheck %s --check-prefix=ALL --check-prefix=SLOW32

define <4 x float> @merge_2_floats(float* nocapture %p) nounwind readonly {
  %tmp1 = load float* %p
  %vecins = insertelement <4 x float> undef, float %tmp1, i32 0
  %add.ptr = getelementptr float* %p, i32 1
  %tmp5 = load float* %add.ptr
  %vecins7 = insertelement <4 x float> %vecins, float %tmp5, i32 1
  ret <4 x float> %vecins7

; ALL-LABEL: merge_2_floats
; ALL: vmovq
; ALL-NEXT: retq
}

define <4 x float> @merge_4_floats(float* %ptr) {
  %a = load float* %ptr, align 8
  %vec = insertelement <4 x float> undef, float %a, i32 0
  %idx1 = getelementptr inbounds float* %ptr, i64 1
  %b = load float* %idx1, align 8
  %vec2 = insertelement <4 x float> %vec, float %b, i32 1
  %idx3 = getelementptr inbounds float* %ptr, i64 2
  %c = load float* %idx3, align 8
  %vec4 = insertelement <4 x float> %vec2, float %c, i32 2
  %idx5 = getelementptr inbounds float* %ptr, i64 3
  %d = load float* %idx5, align 8
  %vec6 = insertelement <4 x float> %vec4, float %d, i32 3
  ret <4 x float> %vec6

; ALL-LABEL: merge_4_floats
; ALL: vmovups
; ALL-NEXT: retq
}

; PR21710 ( http://llvm.org/bugs/show_bug.cgi?id=21710 ) 
; Make sure that 32-byte vectors are handled efficiently.
; If the target has slow 32-byte accesses, we should still generate
; 16-byte loads.

define <8 x float> @merge_8_floats(float* %ptr) {
  %a = load float* %ptr, align 4
  %vec = insertelement <8 x float> undef, float %a, i32 0
  %idx1 = getelementptr inbounds float* %ptr, i64 1
  %b = load float* %idx1, align 4
  %vec2 = insertelement <8 x float> %vec, float %b, i32 1
  %idx3 = getelementptr inbounds float* %ptr, i64 2
  %c = load float* %idx3, align 4
  %vec4 = insertelement <8 x float> %vec2, float %c, i32 2
  %idx5 = getelementptr inbounds float* %ptr, i64 3
  %d = load float* %idx5, align 4
  %vec6 = insertelement <8 x float> %vec4, float %d, i32 3
  %idx7 = getelementptr inbounds float* %ptr, i64 4
  %e = load float* %idx7, align 4
  %vec8 = insertelement <8 x float> %vec6, float %e, i32 4
  %idx9 = getelementptr inbounds float* %ptr, i64 5
  %f = load float* %idx9, align 4
  %vec10 = insertelement <8 x float> %vec8, float %f, i32 5
  %idx11 = getelementptr inbounds float* %ptr, i64 6
  %g = load float* %idx11, align 4
  %vec12 = insertelement <8 x float> %vec10, float %g, i32 6
  %idx13 = getelementptr inbounds float* %ptr, i64 7
  %h = load float* %idx13, align 4
  %vec14 = insertelement <8 x float> %vec12, float %h, i32 7
  ret <8 x float> %vec14

; ALL-LABEL: merge_8_floats

; FAST32: vmovups
; FAST32-NEXT: retq

; SLOW32: vmovups
; SLOW32: vinsertf128
; SLOW32-NEXT: retq
}

define <4 x double> @merge_4_doubles(double* %ptr) {
  %a = load double* %ptr, align 8
  %vec = insertelement <4 x double> undef, double %a, i32 0
  %idx1 = getelementptr inbounds double* %ptr, i64 1
  %b = load double* %idx1, align 8
  %vec2 = insertelement <4 x double> %vec, double %b, i32 1
  %idx3 = getelementptr inbounds double* %ptr, i64 2
  %c = load double* %idx3, align 8
  %vec4 = insertelement <4 x double> %vec2, double %c, i32 2
  %idx5 = getelementptr inbounds double* %ptr, i64 3
  %d = load double* %idx5, align 8
  %vec6 = insertelement <4 x double> %vec4, double %d, i32 3
  ret <4 x double> %vec6

; ALL-LABEL: merge_4_doubles
; FAST32: vmovups
; FAST32-NEXT: retq

; SLOW32: vmovups
; SLOW32: vinsertf128
; SLOW32-NEXT: retq
}

