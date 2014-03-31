; RUN: llc -march=arm64 < %s | FileCheck %s

define float @fma32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fma32:
; CHECK: fmadd
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %0
}

define float @fnma32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnma32:
; CHECK: fnmadd
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %c)
  %mul = fmul float %0, -1.000000e+00
  ret float %mul
}

define float @fms32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms32:
; CHECK: fmsub
  %mul = fmul float %b, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %mul, float %c)
  ret float %0
}

define float @fms32_com(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms32_com:
; CHECK: fmsub
  %mul = fmul float %b, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %mul, float %a, float %c)
  ret float %0
}

define float @fnms32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnms32:
; CHECK: fnmsub
  %mul = fmul float %c, -1.000000e+00
  %0 = tail call float @llvm.fma.f32(float %a, float %b, float %mul)
  ret float %0
}

define double @fma64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fma64:
; CHECK: fmadd
entry:
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %0
}

define double @fnma64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fnma64:
; CHECK: fnmadd
entry:
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %c)
  %mul = fmul double %0, -1.000000e+00
  ret double %mul
}

define double @fms64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fms64:
; CHECK: fmsub
entry:
  %mul = fmul double %b, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %mul, double %c)
  ret double %0
}

define double @fms64_com(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fms64_com:
; CHECK: fmsub
entry:
  %mul = fmul double %b, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %mul, double %a, double %c)
  ret double %0
}

define double @fnms64(double %a, double %b, double %c) nounwind readnone ssp {
; CHECK-LABEL: fnms64:
; CHECK: fnmsub
entry:
  %mul = fmul double %c, -1.000000e+00
  %0 = tail call double @llvm.fma.f64(double %a, double %b, double %mul)
  ret double %0
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
