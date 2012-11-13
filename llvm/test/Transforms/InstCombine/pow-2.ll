; Test that the pow library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

declare float @pow(double, double)

; Check that pow functions with the wrong prototype aren't simplified.

define float @test_no_simplify1(double %x) {
; CHECK: @test_no_simplify1
  %retval = call float @pow(double 1.0, double %x)
; CHECK-NEXT: call float @pow(double 1.000000e+00, double %x)
  ret float %retval
}
