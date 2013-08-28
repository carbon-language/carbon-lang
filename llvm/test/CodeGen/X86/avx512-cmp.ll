; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s

; CHECK: vucomisdz
define double @test1(double %a, double %b) nounwind {
  %tobool = fcmp une double %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub double %a, %b
  ret double %c
l2:
  %c1 = fadd double %a, %b
  ret double %c1
}

; CHECK: vucomissz
define float @test2(float %a, float %b) nounwind {
  %tobool = fcmp olt float %a, %b
  br i1 %tobool, label %l1, label %l2

l1:
  %c = fsub float %a, %b
  ret float %c
l2:
  %c1 = fadd float %a, %b
  ret float %c1
}
