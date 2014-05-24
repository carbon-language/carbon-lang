; RUN: llc -march=arm64 -enable-no-nans-fp-math < %s | FileCheck %s

define double @test_direct(float %in) #1 {
; CHECK-LABEL: test_direct:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double 0.000000e+00, double %longer
  ret double %val

; CHECK: fmax
}

define double @test_cross(float %in) #1 {
; CHECK-LABEL: test_cross:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double %longer, double 0.000000e+00
  ret double %val

; CHECK: fmin
}

; This isn't a min or a max, but passes the first condition for swapping the
; results. Make sure they're put back before we resort to the normal fcsel.
define float @test_cross_fail(float %lhs, float %rhs) {
; CHECK-LABEL: test_cross_fail:
  %tst = fcmp une float %lhs, %rhs
  %res = select i1 %tst, float %rhs, float %lhs
  ret float %res

  ; The register allocator would have to decide to be deliberately obtuse before
  ; other register were used.
; CHECK: fcsel s0, s1, s0, ne
}