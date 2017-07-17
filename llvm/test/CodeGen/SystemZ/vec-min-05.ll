; Test vector minimum on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @fmin(double, double)
declare double @llvm.minnum.f64(double, double)
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>)

declare float @fminf(float, float)
declare float @llvm.minnum.f32(float, float)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)

; Test the fmin library function.
define double @f1(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f1:
; CHECK: wfmindb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @fmin(double %val1, double %val2) readnone
  ret double %ret
}

; Test the f64 minnum intrinsic.
define double @f2(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f2:
; CHECK: wfmindb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @llvm.minnum.f64(double %val1, double %val2)
  ret double %ret
}

; Test a f64 constant compare/select resulting in minnum.
define double @f3(double %dummy, double %val) {
; CHECK-LABEL: f3:
; CHECK: lzdr [[REG:%f[0-9]+]]
; CHECK: wfmindb %f0, %f2, [[REG]], 4
; CHECK: br %r14
  %cmp = fcmp olt double %val, 0.0
  %ret = select i1 %cmp, double %val, double 0.0
  ret double %ret
}

; Test a f64 constant compare/select resulting in minnan.
define double @f4(double %dummy, double %val) {
; CHECK-LABEL: f4:
; CHECK: lzdr [[REG:%f[0-9]+]]
; CHECK: wfmindb %f0, %f2, [[REG]], 1
; CHECK: br %r14
  %cmp = fcmp ult double %val, 0.0
  %ret = select i1 %cmp, double %val, double 0.0
  ret double %ret
}

; Test the v2f64 minnum intrinsic.
define <2 x double> @f5(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f5:
; CHECK: vfmindb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <2 x double> @llvm.minnum.v2f64(<2 x double> %val1, <2 x double> %val2)
  ret <2 x double> %ret
}

; Test the fminf library function.
define float @f11(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f11:
; CHECK: wfminsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @fminf(float %val1, float %val2) readnone
  ret float %ret
}

; Test the f32 minnum intrinsic.
define float @f12(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f12:
; CHECK: wfminsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @llvm.minnum.f32(float %val1, float %val2)
  ret float %ret
}

; Test a f32 constant compare/select resulting in minnum.
define float @f13(float %dummy, float %val) {
; CHECK-LABEL: f13:
; CHECK: lzer [[REG:%f[0-9]+]]
; CHECK: wfminsb %f0, %f2, [[REG]], 4
; CHECK: br %r14
  %cmp = fcmp olt float %val, 0.0
  %ret = select i1 %cmp, float %val, float 0.0
  ret float %ret
}

; Test a f32 constant compare/select resulting in minnan.
define float @f14(float %dummy, float %val) {
; CHECK-LABEL: f14:
; CHECK: lzer [[REG:%f[0-9]+]]
; CHECK: wfminsb %f0, %f2, [[REG]], 1
; CHECK: br %r14
  %cmp = fcmp ult float %val, 0.0
  %ret = select i1 %cmp, float %val, float 0.0
  ret float %ret
}

; Test the v4f32 minnum intrinsic.
define <4 x float> @f15(<4 x float> %dummy, <4 x float> %val1,
                        <4 x float> %val2) {
; CHECK-LABEL: f15:
; CHECK: vfminsb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <4 x float> @llvm.minnum.v4f32(<4 x float> %val1, <4 x float> %val2)
  ret <4 x float> %ret
}

