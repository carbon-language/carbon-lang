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

define i1 @test5(float %a) nounwind {
  %neg = fsub float -0.000000e+00, %a
  %cmp = fcmp ogt float %neg, 1.000000e+00
  ret i1 %cmp
; CHECK: @test5
; CHECK-NEXT: fcmp olt float %a, -1.0
}

define i1 @test6(float %x, float %y) nounwind {
  %neg1 = fsub float -0.000000e+00, %x
  %neg2 = fsub float -0.000000e+00, %y
  %cmp = fcmp olt float %neg1, %neg2
  ret i1 %cmp
; CHECK: @test6
; CHECK-NEXT: fcmp ogt float %x, %y
}

define i1 @test7(float %x) nounwind readnone ssp noredzone {
  %ext = fpext float %x to ppc_fp128
  %cmp = fcmp ogt ppc_fp128 %ext, 0xM00000000000000000000000000000000
  ret i1 %cmp
; Can't convert ppc_fp128
; CHECK: @test7
; CHECK-NEXT: fpext float %x to ppc_fp128
}

define float @test8(float %x) nounwind readnone optsize ssp {
  %conv = fpext float %x to double
  %cmp = fcmp olt double %conv, 0.000000e+00
  %conv1 = zext i1 %cmp to i32
  %conv2 = sitofp i32 %conv1 to float
  ret float %conv2
; Float comparison to zero shouldn't cast to double.
; CHECK: @test8
; CHECK-NEXT: fcmp olt float %x, 0.000000e+00
}
