; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=nehalem | FileCheck %s

define double @test1(double %a, double %b, double %eps) {
  %cmp = fcmp olt double %a, %eps
  %cond = select i1 %cmp, double %b, double 0.000000e+00
  ret double %cond

; CHECK-LABEL: @test1
; CHECK:	cmpltsd	%xmm2, %xmm0
; CHECK-NEXT:	andpd	%xmm1, %xmm0
}

define double @test2(double %a, double %b, double %eps) {
  %cmp = fcmp ole double %a, %eps
  %cond = select i1 %cmp, double %b, double 0.000000e+00
  ret double %cond

; CHECK-LABEL: @test2
; CHECK:	cmplesd	%xmm2, %xmm0
; CHECK-NEXT:	andpd	%xmm1, %xmm0
}

define double @test3(double %a, double %b, double %eps) {
  %cmp = fcmp ogt double %a, %eps
  %cond = select i1 %cmp, double %b, double 0.000000e+00
  ret double %cond

; CHECK-LABEL: @test3
; CHECK:	cmpltsd	%xmm0, %xmm2
; CHECK-NEXT:	andpd	%xmm1, %xmm2
}

define double @test4(double %a, double %b, double %eps) {
  %cmp = fcmp oge double %a, %eps
  %cond = select i1 %cmp, double %b, double 0.000000e+00
  ret double %cond

; CHECK-LABEL: @test4
; CHECK:	cmplesd	%xmm0, %xmm2
; CHECK-NEXT:	andpd	%xmm1, %xmm2
}

define double @test5(double %a, double %b, double %eps) {
  %cmp = fcmp olt double %a, %eps
  %cond = select i1 %cmp, double 0.000000e+00, double %b
  ret double %cond

; CHECK-LABEL: @test5
; CHECK:	cmpltsd	%xmm2, %xmm0
; CHECK-NEXT:	andnpd	%xmm1, %xmm0
}

define double @test6(double %a, double %b, double %eps) {
  %cmp = fcmp ole double %a, %eps
  %cond = select i1 %cmp, double 0.000000e+00, double %b
  ret double %cond

; CHECK-LABEL: @test6
; CHECK:	cmplesd	%xmm2, %xmm0
; CHECK-NEXT:	andnpd	%xmm1, %xmm0
}

define double @test7(double %a, double %b, double %eps) {
  %cmp = fcmp ogt double %a, %eps
  %cond = select i1 %cmp, double 0.000000e+00, double %b
  ret double %cond

; CHECK-LABEL: @test7
; CHECK:	cmpltsd	%xmm0, %xmm2
; CHECK-NEXT:	andnpd	%xmm1, %xmm2
}

define double @test8(double %a, double %b, double %eps) {
  %cmp = fcmp oge double %a, %eps
  %cond = select i1 %cmp, double 0.000000e+00, double %b
  ret double %cond

; CHECK-LABEL: @test8
; CHECK:	cmplesd	%xmm0, %xmm2
; CHECK-NEXT:	andnpd	%xmm1, %xmm2
}

define float @test9(float %a, float %b, float %eps) {
  %cmp = fcmp olt float %a, %eps
  %cond = select i1 %cmp, float %b, float 0.000000e+00
  ret float %cond

; CHECK-LABEL: @test9
; CHECK:	cmpltss	%xmm2, %xmm0
; CHECK-NEXT:	andps	%xmm1, %xmm0
}

define float @test10(float %a, float %b, float %eps) {
  %cmp = fcmp ole float %a, %eps
  %cond = select i1 %cmp, float %b, float 0.000000e+00
  ret float %cond

; CHECK-LABEL: @test10
; CHECK:	cmpless	%xmm2, %xmm0
; CHECK-NEXT:	andps	%xmm1, %xmm0
}

define float @test11(float %a, float %b, float %eps) {
  %cmp = fcmp ogt float %a, %eps
  %cond = select i1 %cmp, float %b, float 0.000000e+00
  ret float %cond

; CHECK-LABEL: @test11
; CHECK:	cmpltss	%xmm0, %xmm2
; CHECK-NEXT:	andps	%xmm1, %xmm2
}

define float @test12(float %a, float %b, float %eps) {
  %cmp = fcmp oge float %a, %eps
  %cond = select i1 %cmp, float %b, float 0.000000e+00
  ret float %cond

; CHECK-LABEL: @test12
; CHECK:	cmpless	%xmm0, %xmm2
; CHECK-NEXT:	andps	%xmm1, %xmm2
}

define float @test13(float %a, float %b, float %eps) {
  %cmp = fcmp olt float %a, %eps
  %cond = select i1 %cmp, float 0.000000e+00, float %b
  ret float %cond

; CHECK-LABEL: @test13
; CHECK:	cmpltss	%xmm2, %xmm0
; CHECK-NEXT:	andnps	%xmm1, %xmm0
}

define float @test14(float %a, float %b, float %eps) {
  %cmp = fcmp ole float %a, %eps
  %cond = select i1 %cmp, float 0.000000e+00, float %b
  ret float %cond

; CHECK-LABEL: @test14
; CHECK:	cmpless	%xmm2, %xmm0
; CHECK-NEXT:	andnps	%xmm1, %xmm0
}

define float @test15(float %a, float %b, float %eps) {
  %cmp = fcmp ogt float %a, %eps
  %cond = select i1 %cmp, float 0.000000e+00, float %b
  ret float %cond

; CHECK-LABEL: @test15
; CHECK:	cmpltss	%xmm0, %xmm2
; CHECK-NEXT:	andnps	%xmm1, %xmm2
}

define float @test16(float %a, float %b, float %eps) {
  %cmp = fcmp oge float %a, %eps
  %cond = select i1 %cmp, float 0.000000e+00, float %b
  ret float %cond

; CHECK-LABEL: @test16
; CHECK:	cmpless	%xmm0, %xmm2
; CHECK-NEXT:	andnps	%xmm1, %xmm2
}

define float @test17(float %a, float %b, float %c, float %eps) {
  %cmp = fcmp oge float %a, %eps
  %cond = select i1 %cmp, float %c, float %b
  ret float %cond

; CHECK-LABEL: @test17
; CHECK: cmpless	%xmm0, %xmm3
; CHECK-NEXT: andps	%xmm3, %xmm2
; CHECK-NEXT: andnps	%xmm1, %xmm3
; CHECK-NEXT: orps	%xmm2, %xmm3
}

define double @test18(double %a, double %b, double %c, double %eps) {
  %cmp = fcmp oge double %a, %eps
  %cond = select i1 %cmp, double %c, double %b
  ret double %cond

; CHECK-LABEL: @test18
; CHECK: cmplesd	%xmm0, %xmm3
; CHECK-NEXT: andpd	%xmm3, %xmm2
; CHECK-NEXT: andnpd	%xmm1, %xmm3
; CHECK-NEXT: orpd	%xmm2, %xmm3
}
