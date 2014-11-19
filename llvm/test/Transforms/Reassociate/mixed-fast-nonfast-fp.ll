; RUN: opt -reassociate %s -S | FileCheck %s

define float @foo(float %a,float %b, float %c) {
; CHECK: %mul3 = fmul float %a, %b
; CHECK-NEXT: fmul fast float %c, 2.000000e+00
; CHECK-NEXT: fadd fast float %factor, %b
; CHECK-NEXT: fmul fast float %tmp1, %a
; CHECK-NEXT: fadd fast float %tmp2, %mul3
; CHECK-NEXT: ret float
  %mul1 = fmul fast float %a, %c
  %mul2 = fmul fast float %a, %b
  %mul3 = fmul float %a, %b
  %mul4 = fmul fast float %a, %c
  %add1 = fadd fast  float %mul1, %mul3
  %add2 = fadd  fast float %mul4, %mul2
  %add3 = fadd fast float %add1, %add2
  ret float %add3
}
