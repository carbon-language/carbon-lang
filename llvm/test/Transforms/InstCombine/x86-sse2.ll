; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define double @test_sqrt_sd_0(double %a) {
; CHECK-LABEL: @test_sqrt_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
; CHECK-NEXT: %3 = tail call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %2)
; CHECK-NEXT: %4 = extractelement <2 x double> %3, i32 0
; CHECK-NEXT: ret double %4
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = tail call <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double> %2)
  %4 = extractelement <2 x double> %3, i32 0
  ret double %4
}

define double @test_add_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_add_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
; CHECK-NEXT: %3 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
; CHECK-NEXT: %5 = tail call <2 x double> @llvm.x86.sse2.add.sd(<2 x double> %2, <2 x double> %4)
; CHECK-NEXT: %6 = extractelement <2 x double> %5, i32 0
; CHECK-NEXT: ret double %6
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.add.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_sub_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_sub_sd_0
; CHECK-NEXT: %1 = fsub double %a, %b
; CHECK-NEXT: ret double %1
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.sub.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_mul_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_mul_sd_0
; CHECK-NEXT: %1 = fmul double %a, %b
; CHECK-NEXT: ret double %1
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.mul.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_div_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_div_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
; CHECK-NEXT: %3 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
; CHECK-NEXT: %5 = tail call <2 x double> @llvm.x86.sse2.div.sd(<2 x double> %2, <2 x double> %4)
; CHECK-NEXT: %6 = extractelement <2 x double> %5, i32 0
; CHECK-NEXT: ret double %6
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.div.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_min_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_min_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %3 = tail call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %1, <2 x double> %2)
; CHECK-NEXT: %4 = extractelement <2 x double> %3, i32 0
; CHECK-NEXT: ret double %4
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.min.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_max_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_max_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %3 = tail call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %1, <2 x double> %2)
; CHECK-NEXT: %4 = extractelement <2 x double> %3, i32 0
; CHECK-NEXT: ret double %4
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.max.sd(<2 x double> %2, <2 x double> %4)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6 
}

define double @test_cmp_sd_0(double %a, double %b) {
; CHECK-LABEL: @test_cmp_sd_0
; CHECK-NEXT: %1 = insertelement <2 x double> undef, double %a, i32 0
; CHECK-NEXT: %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
; CHECK-NEXT: %3 = insertelement <2 x double> undef, double %b, i32 0
; CHECK-NEXT: %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
; CHECK-NEXT: %5 = tail call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %2, <2 x double> %4, i8 0)
; CHECK-NEXT: %6 = extractelement <2 x double> %5, i32 0
; CHECK-NEXT: ret double %6
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 1.000000e+00, i32 1
  %3 = insertelement <2 x double> undef, double %b, i32 0
  %4 = insertelement <2 x double> %3, double 2.000000e+00, i32 1
  %5 = tail call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %2, <2 x double> %4, i8 0)
  %6 = extractelement <2 x double> %5, i32 0
  ret double %6
}

declare <2 x double> @llvm.x86.sse2.sqrt.sd(<2 x double>) nounwind readnone

declare <2 x double> @llvm.x86.sse2.add.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.sub.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.mul.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.div.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.min.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.max.sd(<2 x double>, <2 x double>)
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8)
