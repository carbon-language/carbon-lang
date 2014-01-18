; RUN: opt < %s -instcombine -S | FileCheck %s

; testing-case "float fold(float a) { return 1.2f * a * 2.3f; }"
; 1.2f and 2.3f is supposed to be fold.
define float @fold(float %a) {
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
; CHECK-LABEL: @fold(
; CHECK: fmul fast float %a, 0x4006147AE0000000
}

; Same testing-case as the one used in fold() except that the operators have
; fixed FP mode.
define float @notfold(float %a) {
; CHECK-LABEL: @notfold(
; CHECK: %mul = fmul fast float %a, 0x3FF3333340000000
  %mul = fmul fast float %a, 0x3FF3333340000000
  %mul1 = fmul float %mul, 0x4002666660000000
  ret float %mul1
}

define float @fold2(float %a) {
; CHECK-LABEL: @fold2(
; CHECK: fmul fast float %a, 0x4006147AE0000000
  %mul = fmul float %a, 0x3FF3333340000000
  %mul1 = fmul fast float %mul, 0x4002666660000000
  ret float %mul1
}

; C * f1 + f1 = (C+1) * f1
define double @fold3(double %f1) {
  %t1 = fmul fast double 2.000000e+00, %f1
  %t2 = fadd fast double %f1, %t1
  ret double %t2
; CHECK-LABEL: @fold3(
; CHECK: fmul fast double %f1, 3.000000e+00
}

; (C1 - X) + (C2 - Y) => (C1+C2) - (X + Y)
define float @fold4(float %f1, float %f2) {
  %sub = fsub float 4.000000e+00, %f1
  %sub1 = fsub float 5.000000e+00, %f2
  %add = fadd fast float %sub, %sub1
  ret float %add
; CHECK-LABEL: @fold4(
; CHECK: %1 = fadd fast float %f1, %f2
; CHECK: fsub fast float 9.000000e+00, %1
}

; (X + C1) + C2 => X + (C1 + C2)
define float @fold5(float %f1, float %f2) {
  %add = fadd float %f1, 4.000000e+00
  %add1 = fadd fast float %add, 5.000000e+00
  ret float %add1
; CHECK-LABEL: @fold5(
; CHECK: fadd fast float %f1, 9.000000e+00
}

; (X + X) + X => 3.0 * X
define float @fold6(float %f1) {
  %t1 = fadd fast float %f1, %f1
  %t2 = fadd fast float %f1, %t1
  ret float %t2
; CHECK-LABEL: @fold6(
; CHECK: fmul fast float %f1, 3.000000e+00
}

; C1 * X + (X + X) = (C1 + 2) * X
define float @fold7(float %f1) {
  %t1 = fmul fast float %f1, 5.000000e+00
  %t2 = fadd fast float %f1, %f1
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fold7(
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

; CHECK-LABEL: @fold9(
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
; CHECK-LABEL: @fold10(
; CHECK: %t3 = fadd fast float %t2, -1.000000e+00
; CHECK: ret float %t3
}

; once cause Crash/miscompilation
define float @fail1(float %f1, float %f2) {
  %conv3 = fadd fast float %f1, -1.000000e+00
  %add = fadd fast float %conv3, %conv3
  %add2 = fadd fast float %add, %conv3
  ret float %add2
; CHECK-LABEL: @fail1(
; CHECK: ret
}

define double @fail2(double %f1, double %f2) {
  %t1 = fsub fast double %f1, %f2
  %t2 = fadd fast double %f1, %f2
  %t3 = fsub fast double %t1, %t2
  ret double %t3
; CHECK-LABEL: @fail2(
; CHECK: ret
}

; c1 * x - x => (c1 - 1.0) * x
define float @fold13(float %x) {
  %mul = fmul fast float %x, 7.000000e+00
  %sub = fsub fast float %mul, %x
  ret float %sub
; CHECK: fold13
; CHECK: fmul fast float %x, 6.000000e+00
; CHECK: ret
}

; -x + y => y - x
define float @fold14(float %x, float %y) {
  %neg = fsub fast float -0.0, %x
  %add = fadd fast float %neg, %y
  ret float %add
; CHECK: fold14
; CHECK: fsub fast float %y, %x
; CHECK: ret
}

; x + -y => x - y
define float @fold15(float %x, float %y) {
  %neg = fsub fast float -0.0, %y
  %add = fadd fast float %x, %neg
  ret float %add
; CHECK: fold15
; CHECK: fsub fast float %x, %y
; CHECK: ret
}

; (select X+Y, X-Y) => X + (select Y, -Y)
define float @fold16(float %x, float %y) {
  %cmp = fcmp ogt float %x, %y
  %plus = fadd fast float %x, %y
  %minus = fsub fast float %x, %y
  %r = select i1 %cmp, float %plus, float %minus
  ret float %r
; CHECK: fold16
; CHECK: fsub fast float
; CHECK: select
; CHECK: fadd fast float
; CHECK: ret
}



; =========================================================================
;
;   Testing-cases about fmul begin
;
; =========================================================================

; ((X*C1) + C2) * C3 => (X * (C1*C3)) + (C2*C3) (i.e. distribution)
define float @fmul_distribute1(float %f1) {
  %t1 = fmul float %f1, 6.0e+3
  %t2 = fadd float %t1, 2.0e+3
  %t3 = fmul fast float %t2, 5.0e+3
  ret float %t3
; CHECK-LABEL: @fmul_distribute1(
; CHECK: %1 = fmul fast float %f1, 3.000000e+07
; CHECK: %t3 = fadd fast float %1, 1.000000e+07
}

; (X/C1 + C2) * C3 => X/(C1/C3) + C2*C3
define double @fmul_distribute2(double %f1, double %f2) {
  %t1 = fdiv double %f1, 3.0e+0
  %t2 = fadd double %t1, 5.0e+1
  ; 0x10000000000000 = DBL_MIN
  %t3 = fmul fast double %t2, 0x10000000000000
  ret double %t3

; CHECK-LABEL: @fmul_distribute2(
; CHECK: %1 = fdiv fast double %f1, 0x7FE8000000000000
; CHECK: fadd fast double %1, 0x69000000000000
}

; 5.0e-1 * DBL_MIN yields denormal, so "(f1*3.0 + 5.0e-1) * DBL_MIN" cannot
; be simplified into f1 * (3.0*DBL_MIN) + (5.0e-1*DBL_MIN)
define double @fmul_distribute3(double %f1) {
  %t1 = fdiv double %f1, 3.0e+0
  %t2 = fadd double %t1, 5.0e-1
  %t3 = fmul fast double %t2, 0x10000000000000
  ret double %t3

; CHECK-LABEL: @fmul_distribute3(
; CHECK: fmul fast double %t2, 0x10000000000000
}

; ((X*C1) + C2) * C3 => (X * (C1*C3)) + (C2*C3) (i.e. distribution)
define float @fmul_distribute4(float %f1) {
  %t1 = fmul float %f1, 6.0e+3
  %t2 = fsub float 2.0e+3, %t1
  %t3 = fmul fast float %t2, 5.0e+3
  ret float %t3
; CHECK-LABEL: @fmul_distribute4(
; CHECK: %1 = fmul fast float %f1, 3.000000e+07
; CHECK: %t3 = fsub fast float 1.000000e+07, %1
}

; C1/X * C2 => (C1*C2) / X
define float @fmul2(float %f1) {
  %t1 = fdiv float 2.0e+3, %f1
  %t3 = fmul fast float %t1, 6.0e+3
  ret float %t3
; CHECK-LABEL: @fmul2(
; CHECK: fdiv fast float 1.200000e+07, %f1
}

; X/C1 * C2 => X * (C2/C1) is disabled if X/C1 has multiple uses
@fmul2_external = external global float
define float @fmul2_disable(float %f1) {
  %div = fdiv fast float 1.000000e+00, %f1 
  store float %div, float* @fmul2_external
  %mul = fmul fast float %div, 2.000000e+00
  ret float %mul
; CHECK-LABEL: @fmul2_disable
; CHECK: store
; CHECK: fmul fast
}

; X/C1 * C2 => X * (C2/C1) (if C2/C1 is normal Fp)
define float @fmul3(float %f1, float %f2) {
  %t1 = fdiv float %f1, 2.0e+3
  %t3 = fmul fast float %t1, 6.0e+3
  ret float %t3
; CHECK-LABEL: @fmul3(
; CHECK: fmul fast float %f1, 3.000000e+00
}

; Rule "X/C1 * C2 => X * (C2/C1) is not applicable if C2/C1 is either a special
; value of a denormal. The 0x3810000000000000 here take value FLT_MIN
;
define float @fmul4(float %f1, float %f2) {
  %t1 = fdiv float %f1, 2.0e+3
  %t3 = fmul fast float %t1, 0x3810000000000000
  ret float %t3
; CHECK-LABEL: @fmul4(
; CHECK: fmul fast float %t1, 0x3810000000000000
}

; X / C1 * C2 => X / (C2/C1) if  C1/C2 is either a special value of a denormal,
;  and C2/C1 is a normal value.
;
define float @fmul5(float %f1, float %f2) {
  %t1 = fdiv float %f1, 3.0e+0
  %t3 = fmul fast float %t1, 0x3810000000000000
  ret float %t3
; CHECK-LABEL: @fmul5(
; CHECK: fdiv fast float %f1, 0x47E8000000000000
}

; (X*Y) * X => (X*X) * Y
define float @fmul6(float %f1, float %f2) {
  %mul = fmul float %f1, %f2
  %mul1 = fmul fast float %mul, %f1
  ret float %mul1
; CHECK-LABEL: @fmul6(
; CHECK: fmul fast float %f1, %f1
}

; "(X*Y) * X => (X*X) * Y" is disabled if "X*Y" has multiple uses
define float @fmul7(float %f1, float %f2) {
  %mul = fmul float %f1, %f2
  %mul1 = fmul fast float %mul, %f1
  %add = fadd float %mul1, %mul
  ret float %add
; CHECK-LABEL: @fmul7(
; CHECK: fmul fast float %mul, %f1
}

; =========================================================================
;
;   Testing-cases about negation
;
; =========================================================================
define float @fneg1(float %f1, float %f2) {
  %sub = fsub float -0.000000e+00, %f1
  %sub1 = fsub nsz float 0.000000e+00, %f2
  %mul = fmul float %sub, %sub1
  ret float %mul
; CHECK-LABEL: @fneg1(
; CHECK: fmul float %f1, %f2
}

; =========================================================================
;
;   Testing-cases about div
;
; =========================================================================

; X/C1 / C2 => X * (1/(C2*C1))
define float @fdiv1(float %x) {
  %div = fdiv float %x, 0x3FF3333340000000
  %div1 = fdiv fast float %div, 0x4002666660000000
  ret float %div1
; 0x3FF3333340000000 = 1.2f
; 0x4002666660000000 = 2.3f
; 0x3FD7303B60000000 = 0.36231884057971014492
; CHECK-LABEL: @fdiv1(
; CHECK: fmul fast float %x, 0x3FD7303B60000000
}

; X*C1 / C2 => X * (C1/C2)
define float @fdiv2(float %x) {
  %mul = fmul float %x, 0x3FF3333340000000
  %div1 = fdiv fast float %mul, 0x4002666660000000
  ret float %div1

; 0x3FF3333340000000 = 1.2f
; 0x4002666660000000 = 2.3f
; 0x3FE0B21660000000 = 0.52173918485641479492
; CHECK-LABEL: @fdiv2(
; CHECK: fmul fast float %x, 0x3FE0B21660000000
}

; "X/C1 / C2 => X * (1/(C2*C1))" is disabled (for now) is C2/C1 is a denormal
;
define float @fdiv3(float %x) {
  %div = fdiv float %x, 0x47EFFFFFE0000000
  %div1 = fdiv fast float %div, 0x4002666660000000
  ret float %div1
; CHECK-LABEL: @fdiv3(
; CHECK: fdiv float %x, 0x47EFFFFFE0000000
}

; "X*C1 / C2 => X * (C1/C2)" is disabled if C1/C2 is a denormal
define float @fdiv4(float %x) {
  %mul = fmul float %x, 0x47EFFFFFE0000000
  %div = fdiv float %mul, 0x3FC99999A0000000
  ret float %div
; CHECK-LABEL: @fdiv4(
; CHECK: fmul float %x, 0x47EFFFFFE0000000
}

; (X/Y)/Z = > X/(Y*Z)
define float @fdiv5(float %f1, float %f2, float %f3) {
  %t1 = fdiv float %f1, %f2
  %t2 = fdiv fast float %t1, %f3
  ret float %t2
; CHECK-LABEL: @fdiv5(
; CHECK: fmul float %f2, %f3
}

; Z/(X/Y) = > (Z*Y)/X
define float @fdiv6(float %f1, float %f2, float %f3) {
  %t1 = fdiv float %f1, %f2
  %t2 = fdiv fast float %f3, %t1
  ret float %t2
; CHECK-LABEL: @fdiv6(
; CHECK: fmul float %f3, %f2
}

; C1/(X*C2) => (C1/C2) / X
define float @fdiv7(float %x) {
  %t1 = fmul float %x, 3.0e0
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv7(
; CHECK: fdiv fast float 5.000000e+00, %x
}

; C1/(X/C2) => (C1*C2) / X
define float @fdiv8(float %x) {
  %t1 = fdiv float %x, 3.0e0
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv8(
; CHECK: fdiv fast float 4.500000e+01, %x
}

; C1/(C2/X) => (C1/C2) * X
define float @fdiv9(float %x) {
  %t1 = fdiv float 3.0e0, %x
  %t2 = fdiv fast float 15.0e0, %t1
  ret float %t2
; CHECK-LABEL: @fdiv9(
; CHECK: fmul fast float %x, 5.000000e+00
}

; =========================================================================
;
;   Testing-cases about factorization
;
; =========================================================================
; x*z + y*z => (x+y) * z
define float @fact_mul1(float %x, float %y, float %z) {
  %t1 = fmul fast float %x, %z
  %t2 = fmul fast float %y, %z
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul1(
; CHECK: fmul fast float %1, %z
}

; z*x + y*z => (x+y) * z
define float @fact_mul2(float %x, float %y, float %z) {
  %t1 = fmul fast float %z, %x
  %t2 = fmul fast float %y, %z
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul2(
; CHECK: fmul fast float %1, %z
}

; z*x - z*y => (x-y) * z
define float @fact_mul3(float %x, float %y, float %z) {
  %t2 = fmul fast float %z, %y
  %t1 = fmul fast float %z, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul3(
; CHECK: fmul fast float %1, %z
}

; x*z - z*y => (x-y) * z
define float @fact_mul4(float %x, float %y, float %z) {
  %t1 = fmul fast float %x, %z
  %t2 = fmul fast float %z, %y
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK-LABEL: @fact_mul4(
; CHECK: fmul fast float %1, %z
}

; x/y + x/z, no xform
define float @fact_div1(float %x, float %y, float %z) {
  %t1 = fdiv fast float %x, %y
  %t2 = fdiv fast float %x, %z
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div1
; CHECK: fadd fast float %t1, %t2
}

; x/y + z/x; no xform
define float @fact_div2(float %x, float %y, float %z) {
  %t1 = fdiv fast float %x, %y
  %t2 = fdiv fast float %z, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div2
; CHECK: fadd fast float %t1, %t2
}

; y/x + z/x => (y+z)/x
define float @fact_div3(float %x, float %y, float %z) {
  %t1 = fdiv fast float %y, %x
  %t2 = fdiv fast float %z, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div3
; CHECK: fdiv fast float %1, %x
}

; y/x - z/x => (y-z)/x
define float @fact_div4(float %x, float %y, float %z) {
  %t1 = fdiv fast float %y, %x
  %t2 = fdiv fast float %z, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK: fact_div4
; CHECK: fdiv fast float %1, %x
}

; y/x - z/x => (y-z)/x is disabled if y-z is denormal.
define float @fact_div5(float %x) {
  %t1 = fdiv fast float 0x3810000000000000, %x
  %t2 = fdiv fast float 0x3800000000000000, %x
  %t3 = fadd fast float %t1, %t2
  ret float %t3
; CHECK: fact_div5
; CHECK: fdiv fast float 0x3818000000000000, %x
}

; y/x - z/x => (y-z)/x is disabled if y-z is denormal.
define float @fact_div6(float %x) {
  %t1 = fdiv fast float 0x3810000000000000, %x
  %t2 = fdiv fast float 0x3800000000000000, %x
  %t3 = fsub fast float %t1, %t2
  ret float %t3
; CHECK: fact_div6
; CHECK: %t3 = fsub fast float %t1, %t2
}
