; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.fma.f64(double %f1, double %f2, double %f3)

define double @f1(double %f1, double %f2, double %acc) {
; CHECK-LABEL: f1:
; CHECK: wfnmadb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call double @llvm.fma.f64 (double %f1, double %f2, double %acc)
  %negres = fsub double -0.0, %res
  ret double %negres
}

define double @f2(double %f1, double %f2, double %acc) {
; CHECK-LABEL: f2:
; CHECK: wfnmsdb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %negacc = fsub double -0.0, %acc
  %res = call double @llvm.fma.f64 (double %f1, double %f2, double %negacc)
  %negres = fsub double -0.0, %res
  ret double %negres
}

