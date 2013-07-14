; Test rint()-like rounding, with non-integer values triggering an
; inexact condition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test f32.
declare float @llvm.rint.f32(float %f)
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: fiebr %f0, 0, %f0
; CHECK: br %r14
  %res = call float @llvm.rint.f32(float %f)
  ret float %res
}

; Test f64.
declare double @llvm.rint.f64(double %f)
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: fidbr %f0, 0, %f0
; CHECK: br %r14
  %res = call double @llvm.rint.f64(double %f)
  ret double %res
}

; Test f128.
declare fp128 @llvm.rint.f128(fp128 %f)
define void @f3(fp128 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: fixbr %f0, 0, %f0
; CHECK: br %r14
  %src = load fp128 *%ptr
  %res = call fp128 @llvm.rint.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}
