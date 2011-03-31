; RUN: opt -S -instcombine < %s | FileCheck %s

define i1 @test1(float %x, float %y) nounwind {
  %ext1 = fpext float %x to double
  %ext2 = fpext float %y to double
  %cmp = fcmp ogt double %ext1, %ext2
  ret i1 %cmp
; CHECK: @test1
; CHECK-NEXT: fcmp ogt float %x, %y
}

