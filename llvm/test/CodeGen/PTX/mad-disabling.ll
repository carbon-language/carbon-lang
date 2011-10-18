; RUN: llc < %s -march=ptx32 -mattr=+ptx20,+sm20 | FileCheck %s -check-prefix=FMA
; RUN: llc < %s -march=ptx32 -mattr=+ptx20,+sm20,+no-fma | FileCheck %s -check-prefix=MUL
; RUN: llc < %s -march=ptx64 -mattr=+ptx20,+sm20 | FileCheck %s -check-prefix=FMA
; RUN: llc < %s -march=ptx64 -mattr=+ptx20,+sm20,+no-fma | FileCheck %s -check-prefix=MUL

define ptx_device float @test_mul_add_f(float %x, float %y, float %z) {
entry:
; FMA: mad.rn.f32
; MUL: mul.rn.f32
; MUL: add.rn.f32
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  ret float %b
}

define ptx_device double @test_mul_add_d(double %x, double %y, double %z) {
entry:
; FMA: mad.rn.f64
; MUL: mul.rn.f64
; MUL: add.rn.f64
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  ret double %b
}
