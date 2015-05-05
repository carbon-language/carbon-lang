; Test v2f64 absolute.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x double> @llvm.fabs.v2f64(<2 x double>)

; Test a plain absolute.
define <2 x double> @f1(<2 x double> %val) {
; CHECK-LABEL: f1:
; CHECK: vflpdb %v24, %v24
; CHECK: br %r14
  %ret = call <2 x double> @llvm.fabs.v2f64(<2 x double> %val)
  ret <2 x double> %ret
}

; Test a negative absolute.
define <2 x double> @f2(<2 x double> %val) {
; CHECK-LABEL: f2:
; CHECK: vflndb %v24, %v24
; CHECK: br %r14
  %abs = call <2 x double> @llvm.fabs.v2f64(<2 x double> %val)
  %ret = fsub <2 x double> <double -0.0, double -0.0>, %abs
  ret <2 x double> %ret
}
