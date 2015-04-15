; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

; Anything more than one division using a single divisor operand
; should be converted into a reciprocal and multiplication.

define float @div1_arcp(float %x, float %y, float %z) #0 {
; CHECK-LABEL: div1_arcp:
; CHECK:       # BB#0:
; CHECK-NEXT:    divss %xmm1, %xmm0
; CHECK-NEXT:    retq
  %div1 = fdiv arcp float %x, %y
  ret float %div1
}

define float @div2_arcp(float %x, float %y, float %z) #0 {
; CHECK-LABEL: div2_arcp:
; CHECK:       # BB#0:
; CHECK-NEXT:    movss {{.*#+}} xmm3 = mem[0],zero,zero,zero
; CHECK-NEXT:    divss %xmm2, %xmm3
; CHECK-NEXT:    mulss %xmm3, %xmm0
; CHECK-NEXT:    mulss %xmm1, %xmm0
; CHECK-NEXT:    mulss %xmm3, %xmm0
; CHECK-NEXT:    retq
  %div1 = fdiv arcp float %x, %z
  %mul = fmul arcp float %div1, %y
  %div2 = fdiv arcp float %mul, %z
  ret float %div2
}

; FIXME: If the backend understands 'arcp', then this attribute is unnecessary.
attributes #0 = { "unsafe-fp-math"="true" }
