; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.experimental.constrained.fma.f64(double %f1, double %f2, double %f3, metadata, metadata)
declare float @llvm.experimental.constrained.fma.f32(float %f1, float %f2, float %f3, metadata, metadata)

define double @f1(double %f1, double %f2, double %acc) #0 {
; CHECK-LABEL: f1:
; CHECK: wfnmadb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.fma.f64 (
                        double %f1, double %f2, double %acc,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %negres = fneg double %res
  ret double %negres
}

define double @f2(double %f1, double %f2, double %acc) #0 {
; CHECK-LABEL: f2:
; CHECK: wfnmsdb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %negacc = fneg double %acc
  %res = call double @llvm.experimental.constrained.fma.f64 (
                        double %f1, double %f2, double %negacc,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %negres = fneg double %res
  ret double %negres
}

define float @f3(float %f1, float %f2, float %acc) #0 {
; CHECK-LABEL: f3:
; CHECK: wfnmasb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.fma.f32 (
                        float %f1, float %f2, float %acc,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %negres = fneg float %res
  ret float %negres
}

define float @f4(float %f1, float %f2, float %acc) #0 {
; CHECK-LABEL: f4:
; CHECK: wfnmssb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %negacc = fneg float %acc
  %res = call float @llvm.experimental.constrained.fma.f32 (
                        float %f1, float %f2, float %negacc,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %negres = fneg float %res
  ret float %negres
}

attributes #0 = { strictfp }
