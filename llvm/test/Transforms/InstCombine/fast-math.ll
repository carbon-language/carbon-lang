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

; C * f1 + f1 = (C+1) * f1
define double @fold3(double %f1) {
  %t1 = fmul fast double 2.000000e+00, %f1
  %t2 = fadd fast double %f1, %t1
  ret double %t2
; CHECK: @fold3
; CHECK: fmul fast double %f1, 3.000000e+00
}

; (C1 - X) + (C2 - Y) => (C1+C2) - (X + Y)
define float @fold4(float %f1, float %f2) {
  %sub = fsub float 4.000000e+00, %f1
  %sub1 = fsub float 5.000000e+00, %f2
  %add = fadd fast float %sub, %sub1
  ret float %add
; CHECK: @fold4
; CHECK: %1 = fadd fast float %f1, %f2
; CHECK: fsub fast float 9.000000e+00, %1
}

; (X + C1) + C2 => X + (C1 + C2)
define float @fold5(float %f1, float %f2) {
  %add = fadd float %f1, 4.000000e+00
  %add1 = fadd fast float %add, 5.000000e+00
  ret float %add1
; CHECK: @fold5
; CHECK: fadd float %f1, 9.000000e+00
}

; (X + X) + X => 3.0 * X
define float @fold6(float %f1) {
  %t1 = fadd fast float %f1, %f1
  %t2 = fadd fast float %f1, %t1
  ret float %t2
; CHECK: @fold6
; CHECK: fmul fast float %f1, 3.000000e+00
}

; C1 * X + (X + X) = (C1 + 2) * X
define float @fold7(float %f1) {
  %t1 = fmul fast float %f1, 5.000000e+00
  %t2 = fadd fast float %f1, %f1
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: @fold7
; CHECK: fmul fast float %f1, 7.000000e+00
}

; (X + X) + (X + X) => 4.0 * X
define float @fold8(float %f1) {
  %t1 = fadd fast float %f1, %f1
  %t2 = fadd fast float %f1, %f1
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fold8
; CHECK: fmul fast float %f1, 4.000000e+00
}

; X - (X + Y) => 0 - Y
define float @fold9(float %f1, float %f2) {
  %t1 = fadd float %f1, %f2
  %t3 = fsub fast float %f1, %t1
  ret float %t3

; CHECK: @fold9
; CHECK: fsub fast float 0.000000e+00, %f2
}

; Let C3 = C1 + C2. (f1 + C1) + (f2 + C2) => (f1 + f2) + C3 instead of
; "(f1 + C3) + f2" or "(f2 + C3) + f1". Placing constant-addend at the 
; top of resulting simplified expression tree may potentially reveal some
; optimization opportunities in the super-expression trees.
; 
define float @fold10(float %f1, float %f2) {
  %t1 = fadd fast float 2.000000e+00, %f1
  %t2 = fsub fast float %f2, 3.000000e+00
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: @fold10
; CHECK: %t3 = fadd float %t2, -1.000000e+00
; CHECK: ret float %t3
}

; once cause Crash/miscompilation
define float @fail1(float %f1, float %f2) {
  %conv3 = fadd fast float %f1, -1.000000e+00
  %add = fadd fast float %conv3, %conv3
  %add2 = fadd fast float %add, %conv3
  ret float %add2
; CHECK: @fail1
; CHECK: ret
}

define double @fail2(double %f1, double %f2) {
  %t1 = fsub fast double %f1, %f2
  %t2 = fadd fast double %f1, %f2
  %t3 = fsub fast double %t1, %t2
  ret double %t3
; CHECK: @fail2
; CHECK: ret
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
