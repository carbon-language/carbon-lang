; Test loads of floating-point zero.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test f32.
define float @f1() {
; CHECK-LABEL: f1:
; CHECK: lzer %f0
; CHECK: br %r14
  ret float 0.0
}

; Test f64.
define double @f2() {
; CHECK-LABEL: f2:
; CHECK: lzdr %f0
; CHECK: br %r14
  ret double 0.0
}

; Test f128.
define void @f3(fp128 *%x) {
; CHECK-LABEL: f3:
; CHECK: lzxr %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  store fp128 0xL00000000000000000000000000000000, fp128 *%x
  ret void
}
