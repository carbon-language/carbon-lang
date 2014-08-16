; RUN: llc -march=x86-64 < %s | FileCheck %s

; CHECK-LABEL: fmul2_f32:
; CHECK: addss %xmm0, %xmm0
define float @fmul2_f32(float %x) {
  %y = fmul float %x, 2.0
  ret float %y
}

; fmul 2.0, x -> fadd x, x for vectors.

; CHECK-LABEL: fmul2_v4f32:
; CHECK: addps %xmm0, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul2_v4f32(<4 x float> %x) {
  %y = fmul <4 x float> %x, <float 2.0, float 2.0, float 2.0, float 2.0>
  ret <4 x float> %y
}

; CHECK-LABEL: constant_fold_fmul_v4f32:
; CHECK: movaps
; CHECK-NEXT: ret
define <4 x float> @constant_fold_fmul_v4f32(<4 x float> %x) {
  %y = fmul <4 x float> <float 4.0, float 4.0, float 4.0, float 4.0>, <float 2.0, float 2.0, float 2.0, float 2.0>
  ret <4 x float> %y
}

; CHECK-LABEL: fmul0_v4f32:
; CHECK: xorps %xmm0, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul0_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 0.0, float 0.0, float 0.0, float 0.0>
  ret <4 x float> %y
}

; CHECK-LABEL: fmul_c2_c4_v4f32:
; CHECK-NOT: addps
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_c2_c4_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 2.0, float 2.0, float 2.0, float 2.0>
  %z = fmul <4 x float> %y, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %z
}

; CHECK-LABEL: fmul_c3_c4_v4f32:
; CHECK-NOT: addps
; CHECK: mulps
; CHECK-NOT: mulps
; CHECK-NEXT: ret
define <4 x float> @fmul_c3_c4_v4f32(<4 x float> %x) #0 {
  %y = fmul <4 x float> %x, <float 3.0, float 3.0, float 3.0, float 3.0>
  %z = fmul <4 x float> %y, <float 4.0, float 4.0, float 4.0, float 4.0>
  ret <4 x float> %z
}

; CHECK-LABEL: fmul_c2_c4_f32:
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: mulss
; CHECK-NEXT: ret
define float @fmul_c2_c4_f32(float %x) #0 {
  %y = fmul float %x, 2.0
  %z = fmul float %y, 4.0
  ret float %z
}

; CHECK-LABEL: fmul_c3_c4_f32:
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: mulss
; CHECK-NET: ret
define float @fmul_c3_c4_f32(float %x) #0 {
  %y = fmul float %x, 3.0
  %z = fmul float %y, 4.0
  ret float %z
}

; CHECK-LABEL: fmul_fneg_fneg_f32:
; CHECK: mulss %xmm1, %xmm0
; CHECK-NEXT: retq
define float @fmul_fneg_fneg_f32(float %x, float %y) {
  %x.neg = fsub float -0.0, %x
  %y.neg = fsub float -0.0, %y
  %mul = fmul float %x.neg, %y.neg
  ret float %mul
}
; CHECK-LABEL: fmul_fneg_fneg_v4f32:
; CHECK: mulps {{%xmm1|\(%rdx\)}}, %xmm0
; CHECK-NEXT: retq
define <4 x float> @fmul_fneg_fneg_v4f32(<4 x float> %x, <4 x float> %y) {
  %x.neg = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %x
  %y.neg = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %y
  %mul = fmul <4 x float> %x.neg, %y.neg
  ret <4 x float> %mul
}

attributes #0 = { "less-precise-fpmad"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "unsafe-fp-math"="true" }
