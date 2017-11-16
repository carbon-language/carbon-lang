; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_half(double %x) {
; CHECK-LABEL: @pow_half(
; CHECK-NEXT:    [[SQRT:%.*]] = call fast double @sqrt(double %x) #1
; CHECK-NEXT:    ret double [[SQRT]]
;
  %pow = call fast double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

define double @pow_neghalf(double %x) {
; CHECK-LABEL: @pow_neghalf(
; CHECK-NEXT:    [[SQRT:%.*]] = call fast double @sqrt(double %x) #1
; CHECK-NEXT:    [[SQRTRECIP:%.*]] = fdiv fast double 1.000000e+00, [[SQRT]]
; CHECK-NEXT:    ret double [[SQRTRECIP]]
;
  %pow = call fast double @llvm.pow.f64(double %x, double -5.000000e-01)
  ret double %pow
}

declare double @llvm.pow.f64(double, double) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
