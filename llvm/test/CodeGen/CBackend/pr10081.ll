; RUN: llc < %s -march=c | grep {static float _ZL3foo} | count 1
; PR10081

@_ZL3foo = internal global float 0.000000e+00, align 4

define float @_Z3barv() nounwind ssp {
  %1 = load float* @_ZL3foo, align 4
  %2 = fadd float %1, 1.000000e+00
  store float %2, float* @_ZL3foo, align 4
  ret float %1
}
