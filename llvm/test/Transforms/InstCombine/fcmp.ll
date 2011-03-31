; RUN: opt -S -instcombine < %s | FileCheck %s

define i1 @test1(float %x, float %y) nounwind {
  %ext1 = fpext float %x to double
  %ext2 = fpext float %y to double
  %cmp = fcmp ogt double %ext1, %ext2
  ret i1 %cmp
; CHECK: @test1
; CHECK-NEXT: fcmp ogt float %x, %y
}

define i1 @test2(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 1.000000e+00
  ret i1 %cmp
; CHECK: @test2
; CHECK-NEXT: fcmp ogt float %a, 1.0
}

define i1 @test3(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 0x3FF0000000000001 ; more precision than float.
  ret i1 %cmp
; CHECK: @test3
; CHECK-NEXT: fpext float %a to double
}

define i1 @test4(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 0x36A0000000000000 ; denormal in float.
  ret i1 %cmp
; CHECK: @test4
; CHECK-NEXT: fpext float %a to double
}
