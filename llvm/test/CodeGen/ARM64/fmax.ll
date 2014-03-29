; RUN: llc -march=arm64 -enable-no-nans-fp-math < %s | FileCheck %s

define double @test_direct(float %in) #1 {
entry:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double 0.000000e+00, double %longer
  ret double %val

; CHECK: fmax
}

define double @test_cross(float %in) #1 {
entry:
  %cmp = fcmp olt float %in, 0.000000e+00
  %longer = fpext float %in to double
  %val = select i1 %cmp, double %longer, double 0.000000e+00
  ret double %val

; CHECK: fmin
}
