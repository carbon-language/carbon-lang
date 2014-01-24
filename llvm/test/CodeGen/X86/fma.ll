; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=+fma,-fma4  | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=i386-apple-darwin10  -mattr=-fma,-fma4  | FileCheck %s --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -mattr=+fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -mtriple=x86_64-apple-darwin10  -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-CALL
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma4  | FileCheck %s --check-prefix=CHECK-FMA-INST
; RUN: llc < %s -march=x86 -mcpu=bdver2 -mattr=-fma,-fma4 | FileCheck %s --check-prefix=CHECK-FMA-CALL

; CHECK: test_f32
; CHECK-FMA-INST: vfmadd213ss
; CHECK-FMA-CALL: fmaf

define float @test_f32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
  %call = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  ret float %call
}

; CHECK: test_f64
; CHECK-FMA-INST: vfmadd213sd
; CHECK-FMA-CALL: fma

define double @test_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
  %call = tail call double @llvm.fma.f64(double %a, double %b, double %c) nounwind readnone
  ret double %call
}

; CHECK: test_f80
; CHECK: fmal

define x86_fp80 @test_f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) nounwind readnone ssp {
entry:
  %call = tail call x86_fp80 @llvm.fma.f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c) nounwind readnone
  ret x86_fp80 %call
}

; CHECK: test_f32_cst
; CHECK-NOT: fma
define float @test_f32_cst() nounwind readnone ssp {
entry:
  %call = tail call float @llvm.fma.f32(float 3.0, float 3.0, float 3.0) nounwind readnone
  ret float %call
}

; Test FMA3 variant selection
; CHECK: fma3_select231ssX:
; CHECK: vfmadd231ss xmm
define float @fma3_select231ssX(float %x, float %y) #0 {
entry:
  br label %while.body
while.body:                                       ; preds = %while.body, %while.body
  %acc.01 = phi float [ 0.000000e+00, %entry ], [ %acc, %while.body ]
  %acc = tail call float @llvm.fma.f32(float %x, float %y, float %acc.01) nounwind readnone
  %b = fcmp ueq float %acc, 0.0
  br i1 %b, label %while.body, label %while.end
while.end:                                        ; preds = %while.body, %entry
  ret float %acc
}

; Test FMA3 variant selection
; CHECK: fma3_select231pdY:
; CHECK: vfmadd231pd ymm
define <4 x double> @fma3_select231pdY(<4 x double> %x, <4 x double> %y) #0 {
entry:
  br label %while.body
while.body:                                       ; preds = %entry, %while.body
  %acc.04 = phi <4 x double> [ zeroinitializer, %entry ], [ %add, %while.body ]
  %add = tail call <4 x double> @llvm.fma.v4f64(<4 x double> %x, <4 x double> %y, <4 x double> %acc.04)
  %vecext = extractelement <4 x double> %add, i32 0
  %cmp = fcmp oeq double %vecext, 0.000000e+00
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body
  ret <4 x double> %add
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare x86_fp80 @llvm.fma.f80(x86_fp80, x86_fp80, x86_fp80) nounwind readnone
declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
