; RUN: opt -S -instcombine < %s | FileCheck %s

declare float @llvm.fma.f32(float, float, float) #0
declare float @llvm.fmuladd.f32(float, float, float) #0

declare double @llvm.fma.f64(double, double, double) #0
declare double @llvm.fmuladd.f64(double, double, double) #0



; CHECK-LABEL: @constant_fold_fma_f32
; CHECK-NEXT: ret float 6.000000e+00
define float @constant_fold_fma_f32() #0 {
  %x = call float @llvm.fma.f32(float 1.0, float 2.0, float 4.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_fmuladd_f32
; CHECK-NEXT: ret float 6.000000e+00
define float @constant_fold_fmuladd_f32() #0 {
  %x = call float @llvm.fmuladd.f32(float 1.0, float 2.0, float 4.0) #0
  ret float %x
}

; CHECK-LABEL: @constant_fold_fma_f64
; CHECK-NEXT: ret double 6.000000e+00
define double @constant_fold_fma_f64() #0 {
  %x = call double @llvm.fma.f64(double 1.0, double 2.0, double 4.0) #0
  ret double %x
}

; CHECK-LABEL: @constant_fold_fmuladd_f64
; CHECK-NEXT: ret double 6.000000e+00
define double @constant_fold_fmuladd_f64() #0 {
  %x = call double @llvm.fmuladd.f64(double 1.0, double 2.0, double 4.0) #0
  ret double %x
}

attributes #0 = { nounwind readnone }
