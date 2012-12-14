; RUN: opt < %s -instcombine -S | FileCheck %s

; testing-case "float fold(float a) { return 1.2f * a * 2.3f; }"
; 1.2f and 2.3f is supposed to be fold.
define float @fold(float %a) {
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
; CHECK: @fold
; CHECK: fmul float %a, 0x4006147AE0000000
}

; Same testing-case as the one used in fold() except that the operators have
; fixed FP mode.
define float @notfold(float %a) {
; CHECK: @notfold
; CHECK: %mul = fmul fast float %a, 0x3FF3333340000000
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul float %mul, 0x4002666660000000
  ret float %mul1
}

define float @fold2(float %a) {
; CHECK: @fold2
; CHECK: fmul float %a, 0x4006147AE0000000
  %mul = fmul float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
}

; rdar://12753946:  x * cond ? 1.0 : 0.0 => cond ? x : 0.0
define double @select1(i32 %cond, double %x, double %y) {
  %tobool = icmp ne i32 %cond, 0
  %cond1 = select i1 %tobool, double 1.000000e+00, double 0.000000e+00
  %mul = fmul nnan nsz double %cond1, %x
  %add = fadd double %mul, %y
  ret double %add
; CHECK: @select1
; CHECK: select i1 %tobool, double %x, double 0.000000e+00
}

define double @select2(i32 %cond, double %x, double %y) {
  %tobool = icmp ne i32 %cond, 0
  %cond1 = select i1 %tobool, double 0.000000e+00, double 1.000000e+00
  %mul = fmul nnan nsz double %cond1, %x
  %add = fadd double %mul, %y
  ret double %add
; CHECK: @select2
; CHECK: select i1 %tobool, double 0.000000e+00, double %x
}

define double @select3(i32 %cond, double %x, double %y) {
  %tobool = icmp ne i32 %cond, 0
  %cond1 = select i1 %tobool, double 0.000000e+00, double 2.000000e+00
  %mul = fmul nnan nsz double %cond1, %x
  %add = fadd double %mul, %y
  ret double %add
; CHECK: @select3
; CHECK: fmul nnan nsz double %cond1, %x
}
