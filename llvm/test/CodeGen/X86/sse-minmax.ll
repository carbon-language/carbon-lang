; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s
; RUN: llc < %s -march=x86-64 -asm-verbose=false -enable-unsafe-fp-math | FileCheck -check-prefix=UNSAFE %s

; Some of these patterns can be matched as SSE min or max. Some of
; then can be matched provided that the operands are swapped.
; Some of them can't be matched at all and require a comparison
; and a conditional branch.

; The naming convention is {,x_}{o,u}{gt,lt,ge,le}{,_inverse}
; x_ : use 0.0 instead of %y
; _inverse : swap the arms of the select.

; CHECK:      ogt:
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ogt:
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ogt(double %x, double %y) nounwind {
  %c = fcmp ogt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      olt:
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      olt:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @olt(double %x, double %y) nounwind {
  %c = fcmp olt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ogt_inverse:
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ogt_inverse:
; UNSAFE-NEXT: minsd  %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ogt_inverse(double %x, double %y) nounwind {
  %c = fcmp ogt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      olt_inverse:
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      olt_inverse:
; UNSAFE-NEXT: maxsd  %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @olt_inverse(double %x, double %y) nounwind {
  %c = fcmp olt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      oge:
; CHECK-NEXT: ucomisd %xmm1, %xmm0
; UNSAFE:      oge:
; UNSAFE-NEXT: maxsd	%xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @oge(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ole:
; CHECK-NEXT: ucomisd %xmm0, %xmm1
; UNSAFE:      ole:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
define double @ole(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      oge_inverse:
; CHECK-NEXT: ucomisd %xmm1, %xmm0
; UNSAFE:      oge_inverse:
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @oge_inverse(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ole_inverse:
; CHECK-NEXT: ucomisd %xmm0, %xmm1
; UNSAFE:      ole_inverse:
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ole_inverse(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      x_ogt:
; CHECK-NEXT: pxor  %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ogt:
; UNSAFE-NEXT: pxor  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ogt(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_olt:
; CHECK-NEXT: pxor  %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_olt:
; UNSAFE-NEXT: pxor  %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_olt(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ogt_inverse:
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ogt_inverse:
; UNSAFE-NEXT: pxor   %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ogt_inverse(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_olt_inverse:
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_olt_inverse:
; UNSAFE-NEXT: pxor   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_olt_inverse(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_oge:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_oge:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_oge(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ole:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ole:
; UNSAFE-NEXT: pxor %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ole(double %x) nounwind {
  %c = fcmp ole double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_oge_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_oge_inverse:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_oge_inverse(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ole_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ole_inverse:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ole_inverse(double %x) nounwind {
  %c = fcmp ole double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      ugt:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      ugt:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ugt(double %x, double %y) nounwind {
  %c = fcmp ugt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ult:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      ult:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ult(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ugt_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      ugt_inverse:
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ugt_inverse(double %x, double %y) nounwind {
  %c = fcmp ugt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ult_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      ult_inverse:
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ult_inverse(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      uge:
; CHECK-NEXT: maxsd   %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      uge:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @uge(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ule:
; CHECK-NEXT: minsd  %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ule:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ule(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      uge_inverse:
; CHECK-NEXT: minsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      uge_inverse:
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @uge_inverse(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ule_inverse:
; CHECK-NEXT: maxsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ule_inverse:
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @ule_inverse(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      x_ugt:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ugt:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ugt(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ult:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_ult:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ult(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ugt_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ugt_inverse:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ugt_inverse(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ult_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_ult_inverse:
; UNSAFE-NEXT: pxor    %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movapd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ult_inverse(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_uge:
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_uge:
; UNSAFE-NEXT: pxor   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_uge(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ule:
; CHECK-NEXT: pxor   %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ule:
; UNSAFE-NEXT: pxor   %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ule(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_uge_inverse:
; CHECK-NEXT: pxor  %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_uge_inverse:
; UNSAFE-NEXT: pxor  %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_uge_inverse(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ule_inverse:
; CHECK-NEXT: pxor  %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ule_inverse:
; UNSAFE-NEXT: pxor  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movapd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define double @x_ule_inverse(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; Test a few more misc. cases.

; CHECK: clampTo3k_a:
; CHECK: minsd
; UNSAFE: clampTo3k_a:
; UNSAFE: minsd
define double @clampTo3k_a(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_b:
; CHECK: minsd
; UNSAFE: clampTo3k_b:
; UNSAFE: minsd
define double @clampTo3k_b(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_c:
; CHECK: maxsd
; UNSAFE: clampTo3k_c:
; UNSAFE: maxsd
define double @clampTo3k_c(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_d:
; CHECK: maxsd
; UNSAFE: clampTo3k_d:
; UNSAFE: maxsd
define double @clampTo3k_d(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_e:
; CHECK: maxsd
; UNSAFE: clampTo3k_e:
; UNSAFE: maxsd
define double @clampTo3k_e(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_f:
; CHECK: maxsd
; UNSAFE: clampTo3k_f:
; UNSAFE: maxsd
define double @clampTo3k_f(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_g:
; CHECK: minsd
; UNSAFE: clampTo3k_g:
; UNSAFE: minsd
define double @clampTo3k_g(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_h:
; CHECK: minsd
; UNSAFE: clampTo3k_h:
; UNSAFE: minsd
define double @clampTo3k_h(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}
