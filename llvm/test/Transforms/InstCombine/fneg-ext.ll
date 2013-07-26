; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK: test1
define double @test1(float %a, double %b) nounwind readnone ssp uwtable {
; CHECK-NOT: fsub
; CHECK: fpext
; CHECK: fadd
  %1 = fsub float -0.000000e+00, %a
  %2 = fpext float %1 to double
  %3 = fsub double %b, %2
  ret double %3
}
