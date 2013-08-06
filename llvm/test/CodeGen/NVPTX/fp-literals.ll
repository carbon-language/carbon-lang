; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Make sure we can properly differentiate between single-precision and
; double-precision FP literals.

; CHECK: myaddf
; CHECK: add.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, 0f3F800000
define float @myaddf(float %a) {
  %ret = fadd float %a, 1.0
  ret float %ret
}

; CHECK: myaddd
; CHECK: add.f64 %fl{{[0-9]+}}, %fl{{[0-9]+}}, 0d3FF0000000000000
define double @myaddd(double %a) {
  %ret = fadd double %a, 1.0
  ret double %ret
}
