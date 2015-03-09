; RUN: opt -S -instcombine < %s | FileCheck %s

declare double @llvm.fabs.f64(double) nounwind readnone

define i1 @test1(float %x, float %y) nounwind {
  %ext1 = fpext float %x to double
  %ext2 = fpext float %y to double
  %cmp = fcmp ogt double %ext1, %ext2
  ret i1 %cmp
; CHECK-LABEL: @test1(
; CHECK-NEXT: fcmp ogt float %x, %y
}

define i1 @test2(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 1.000000e+00
  ret i1 %cmp
; CHECK-LABEL: @test2(
; CHECK-NEXT: fcmp ogt float %a, 1.0
}

define i1 @test3(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 0x3FF0000000000001 ; more precision than float.
  ret i1 %cmp
; CHECK-LABEL: @test3(
; CHECK-NEXT: fpext float %a to double
}

define i1 @test4(float %a) nounwind {
  %ext = fpext float %a to double
  %cmp = fcmp ogt double %ext, 0x36A0000000000000 ; denormal in float.
  ret i1 %cmp
; CHECK-LABEL: @test4(
; CHECK-NEXT: fpext float %a to double
}

define i1 @test5(float %a) nounwind {
  %neg = fsub float -0.000000e+00, %a
  %cmp = fcmp ogt float %neg, 1.000000e+00
  ret i1 %cmp
; CHECK-LABEL: @test5(
; CHECK-NEXT: fcmp olt float %a, -1.0
}

define i1 @test6(float %x, float %y) nounwind {
  %neg1 = fsub float -0.000000e+00, %x
  %neg2 = fsub float -0.000000e+00, %y
  %cmp = fcmp olt float %neg1, %neg2
  ret i1 %cmp
; CHECK-LABEL: @test6(
; CHECK-NEXT: fcmp ogt float %x, %y
}

define i1 @test7(float %x) nounwind readnone ssp noredzone {
  %ext = fpext float %x to ppc_fp128
  %cmp = fcmp ogt ppc_fp128 %ext, 0xM00000000000000000000000000000000
  ret i1 %cmp
; CHECK-LABEL: @test7(
; CHECK-NEXT: fcmp ogt float %x, 0.000000e+00
}

define float @test8(float %x) nounwind readnone optsize ssp {
  %conv = fpext float %x to double
  %cmp = fcmp olt double %conv, 0.000000e+00
  %conv1 = zext i1 %cmp to i32
  %conv2 = sitofp i32 %conv1 to float
  ret float %conv2
; Float comparison to zero shouldn't cast to double.
; CHECK-LABEL: @test8(
; CHECK-NEXT: fcmp olt float %x, 0.000000e+00
}

declare double @fabs(double) nounwind readnone

define i32 @test9(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp olt double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test9(
; CHECK-NOT: fabs
; CHECK: ret i32 0
}

define i32 @test9_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp olt double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test9_intrinsic(
; CHECK-NOT: fabs
; CHECK: ret i32 0
}

define i32 @test10(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp ole double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test10(
; CHECK-NOT: fabs
; CHECK: fcmp oeq double %a, 0.000000e+00
}

define i32 @test10_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp ole double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test10_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp oeq double %a, 0.000000e+00
}

define i32 @test11(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp ogt double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test11(
; CHECK-NOT: fabs
; CHECK: fcmp one double %a, 0.000000e+00
}

define i32 @test11_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp ogt double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test11_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp one double %a, 0.000000e+00
}

define i32 @test12(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp oge double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test12(
; CHECK-NOT: fabs
; CHECK: fcmp ord double %a, 0.000000e+00
}

define i32 @test12_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp oge double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test12_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp ord double %a, 0.000000e+00
}

define i32 @test13(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp une double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test13(
; CHECK-NOT: fabs
; CHECK: fcmp une double %a, 0.000000e+00
}

define i32 @test13_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp une double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test13_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp une double %a, 0.000000e+00
}

define i32 @test14(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp oeq double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test14(
; CHECK-NOT: fabs
; CHECK: fcmp oeq double %a, 0.000000e+00
}

define i32 @test14_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp oeq double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test14_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp oeq double %a, 0.000000e+00
}

define i32 @test15(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp one double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test15(
; CHECK-NOT: fabs
; CHECK: fcmp one double %a, 0.000000e+00
}

define i32 @test15_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp one double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test15_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp one double %a, 0.000000e+00
}

define i32 @test16(double %a) nounwind {
  %call = tail call double @fabs(double %a) nounwind
  %cmp = fcmp ueq double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test16(
; CHECK-NOT: fabs
; CHECK: fcmp ueq double %a, 0.000000e+00
}

define i32 @test16_intrinsic(double %a) nounwind {
  %call = tail call double @llvm.fabs.f64(double %a) nounwind
  %cmp = fcmp ueq double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: @test16_intrinsic(
; CHECK-NOT: fabs
; CHECK: fcmp ueq double %a, 0.000000e+00
}

; Don't crash.
define i32 @test17(double %a, double (double)* %p) nounwind {
  %call = tail call double %p(double %a) nounwind
  %cmp = fcmp ueq double %call, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Can fold fcmp with undef on one side by choosing NaN for the undef
define i32 @test18_undef_unordered(float %a) nounwind {
; CHECK-LABEL: @test18_undef_unordered
; CHECK: ret i32 1
  %cmp = fcmp ueq float %a, undef
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
; Can fold fcmp with undef on one side by choosing NaN for the undef
define i32 @test18_undef_ordered(float %a) nounwind {
; CHECK-LABEL: @test18_undef_ordered
; CHECK: ret i32 0
  %cmp = fcmp oeq float %a, undef
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Can fold fcmp with undef on both side
;   fcmp u_pred undef, undef -> true
;   fcmp o_pred undef, undef -> false
; because whatever you choose for the first undef
; you can choose NaN for the other undef
define i1 @test19_undef_unordered() nounwind {
; CHECK-LABEL: @test19_undef
; CHECK: ret i1 true
  %cmp = fcmp ueq float undef, undef
  ret i1 %cmp
}
define i1 @test19_undef_ordered() nounwind {
; CHECK-LABEL: @test19_undef
; CHECK: ret i1 false
  %cmp = fcmp oeq float undef, undef
  ret i1 %cmp
}
