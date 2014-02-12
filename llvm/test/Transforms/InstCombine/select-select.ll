; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK: @foo1
define float @foo1(float %a) #0 {
; CHECK-NOT: xor
  %b = fcmp ogt float %a, 0.000000e+00
  %c = select i1 %b, float %a, float 0.000000e+00
  %d = fcmp olt float %c, 1.000000e+00
  %f = select i1 %d, float %c, float 1.000000e+00
  ret float %f
}

; CHECK: @foo2
define float @foo2(float %a) #0 {
; CHECK-NOT: xor
  %b = fcmp ogt float %a, 0.000000e+00
  %c = select i1 %b, float %a, float 0.000000e+00
  %d = fcmp olt float %c, 1.000000e+00
  %e = select i1 %b, float %a, float 0.000000e+00
  %f = select i1 %d, float %e, float 1.000000e+00
  ret float %f
}

attributes #0 = { nounwind readnone ssp uwtable }
