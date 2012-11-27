; RUN: opt < %s -instsimplify -S | FileCheck %s

;; x * 0 ==> 0 when no-nans and no-signed-zero
; CHECK: mul_zero_1
define float @mul_zero_1(float %a) {
  %b = fmul nsz nnan float %a, 0.0
; CHECK: ret float 0.0
  ret float %b
}
; CHECK: mul_zero_2
define float @mul_zero_2(float %a) {
  %b = fmul fast float 0.0, %a
; CHECK: ret float 0.0
  ret float %b
}

;; x * 0 =/=> 0 when there could be nans or -0
; CHECK: no_mul_zero_1
define float @no_mul_zero_1(float %a) {
  %b = fmul nsz float %a, 0.0
; CHECK: ret float %b
  ret float %b
}
; CHECK: no_mul_zero_2
define float @no_mul_zero_2(float %a) {
  %b = fmul nnan float %a, 0.0
; CHECK: ret float %b
  ret float %b
}
; CHECK: no_mul_zero_3
define float @no_mul_zero_3(float %a) {
  %b = fmul float %a, 0.0
; CHECK: ret float %b
  ret float %b
}
