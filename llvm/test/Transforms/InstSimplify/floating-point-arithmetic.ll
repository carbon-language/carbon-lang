; RUN: opt < %s -instsimplify -S | FileCheck %s

; fsub 0, (fsub 0, X) ==> X
; CHECK-LABEL: @fsub_0_0_x(
define float @fsub_0_0_x(float %a) {
  %t1 = fsub float -0.0, %a
  %ret = fsub float -0.0, %t1

; CHECK: ret float %a
  ret float %ret
}

; fsub X, 0 ==> X
; CHECK-LABEL: @fsub_x_0(
define float @fsub_x_0(float %a) {
  %ret = fsub float %a, 0.0
; CHECK: ret float %a
  ret float %ret
}

; fadd X, -0 ==> X
; CHECK-LABEL: @fadd_x_n0(
define float @fadd_x_n0(float %a) {
  %ret = fadd float %a, -0.0
; CHECK: ret float %a
  ret float %ret
}

; fmul X, 1.0 ==> X
; CHECK-LABEL: @fmul_X_1(
define double @fmul_X_1(double %a) {
  %b = fmul double 1.000000e+00, %a                ; <double> [#uses=1]
  ; CHECK: ret double %a
  ret double %b
}
