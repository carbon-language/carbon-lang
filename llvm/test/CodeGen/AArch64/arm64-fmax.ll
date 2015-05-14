; RUN: llc -march=arm64 -enable-no-nans-fp-math < %s | FileCheck %s
; RUN: llc -march=arm64 < %s | FileCheck %s --check-prefix=CHECK-SAFE

define double @test_direct(float %in) {
; CHECK-LABEL: test_direct:
; CHECK-SAFE-LABEL: test_direct:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double 0.000000e+00, double %longer
  ret double %val

; CHECK: fmax
; CHECK-SAFE: fmax
}

define double @test_cross(float %in) {
; CHECK-LABEL: test_cross:
; CHECK-SAFE-LABEL: test_cross:
  %cmp = fcmp ult float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double %longer, double 0.000000e+00
  ret double %val

; CHECK: fmin
; CHECK-SAFE: fmin
}

; Same as previous, but with ordered comparison;
; can't be converted in safe-math mode.
define double @test_cross_fail_nan(float %in) {
; CHECK-LABEL: test_cross_fail_nan:
; CHECK-SAFE-LABEL: test_cross_fail_nan:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double %longer, double 0.000000e+00
  ret double %val

; CHECK: fmin
; CHECK-SAFE: fcsel d0, d1, d0, mi
}

; This isn't a min or a max, but passes the first condition for swapping the
; results. Make sure they're put back before we resort to the normal fcsel.
define float @test_cross_fail(float %lhs, float %rhs) {
; CHECK-LABEL: test_cross_fail:
; CHECK-SAFE-LABEL: test_cross_fail:
  %tst = fcmp une float %lhs, %rhs
  %res = select i1 %tst, float %rhs, float %lhs
  ret float %res

  ; The register allocator would have to decide to be deliberately obtuse before
  ; other register were used.
; CHECK: fcsel s0, s1, s0, ne
; CHECK-SAFE: fcsel s0, s1, s0, ne
}

; Make sure the transformation isn't triggered for integers
define i64 @test_integer(i64  %in) {
  %cmp = icmp slt i64 %in, 0
  %val = select i1 %cmp, i64 0, i64 %in
  ret i64 %val
}
