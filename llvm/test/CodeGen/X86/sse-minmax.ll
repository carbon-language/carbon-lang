; RUN: llc < %s -march=x86-64 -asm-verbose=false -join-physregs -promote-elements | FileCheck %s
; RUN: llc < %s -march=x86-64 -asm-verbose=false -join-physregs -enable-unsafe-fp-math -enable-no-nans-fp-math -promote-elements | FileCheck -check-prefix=UNSAFE %s
; RUN: llc < %s -march=x86-64 -asm-verbose=false -join-physregs -enable-no-nans-fp-math -promote-elements | FileCheck -check-prefix=FINITE %s

; Some of these patterns can be matched as SSE min or max. Some of
; then can be matched provided that the operands are swapped.
; Some of them can't be matched at all and require a comparison
; and a conditional branch.

; The naming convention is {,x_,y_}{o,u}{gt,lt,ge,le}{,_inverse}
; x_ : use 0.0 instead of %y
; y_ : use -0.0 instead of %y
; _inverse : swap the arms of the select.

; Some of these tests depend on -join-physregs commuting instructions to
; eliminate copies.

; CHECK:      ogt:
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ogt:
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ogt:
; FINITE-NEXT: maxsd %xmm1, %xmm0
; FINITE-NEXT: ret
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
; FINITE:      olt:
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt(double %x, double %y) nounwind {
  %c = fcmp olt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ogt_inverse:
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ogt_inverse:
; UNSAFE-NEXT: minsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ogt_inverse:
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt_inverse(double %x, double %y) nounwind {
  %c = fcmp ogt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      olt_inverse:
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      olt_inverse:
; UNSAFE-NEXT: maxsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      olt_inverse:
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
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
; FINITE:      oge:
; FINITE-NEXT: maxsd	%xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ole:
; CHECK-NEXT: ucomisd %xmm0, %xmm1
; UNSAFE:      ole:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; FINITE:      ole:
; FINITE-NEXT: minsd %xmm1, %xmm0
define double @ole(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      oge_inverse:
; CHECK-NEXT: ucomisd %xmm1, %xmm0
; UNSAFE:      oge_inverse:
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      oge_inverse:
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge_inverse(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ole_inverse:
; CHECK-NEXT: ucomisd %xmm0, %xmm1
; UNSAFE:      ole_inverse:
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ole_inverse:
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ole_inverse(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      x_ogt:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ogt:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ogt:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: maxsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ogt(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_olt:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_olt:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_olt:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_olt(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ogt_inverse:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ogt_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ogt_inverse:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ogt_inverse(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_olt_inverse:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_olt_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_olt_inverse:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_olt_inverse(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_oge:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_oge:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_oge:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_oge(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ole:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ole:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ole:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ole(double %x) nounwind {
  %c = fcmp ole double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_oge_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_oge_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_oge_inverse:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_oge_inverse(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ole_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ole_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ole_inverse:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
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
; FINITE:      ugt:
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
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
; FINITE:      ult:
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ugt_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      ugt_inverse:
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ugt_inverse:
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt_inverse(double %x, double %y) nounwind {
  %c = fcmp ugt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ult_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      ult_inverse:
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ult_inverse:
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult_inverse(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      uge:
; CHECK-NEXT: maxsd   %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      uge:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      uge:
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      ule:
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ule:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ule:
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK:      uge_inverse:
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      uge_inverse:
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      uge_inverse:
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge_inverse(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      ule_inverse:
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      ule_inverse:
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      ule_inverse:
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule_inverse(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK:      x_ugt:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ugt:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ugt:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ugt(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ult:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_ult:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ult:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ult(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ugt_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      x_ugt_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ugt_inverse:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ugt_inverse(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ult_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      x_ult_inverse:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ult_inverse:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ult_inverse(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_uge:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_uge:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_uge:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: maxsd  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_uge(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_ule:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ule:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ule:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: minsd  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ule(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK:      x_uge_inverse:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_uge_inverse:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_uge_inverse:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_uge_inverse(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      x_ule_inverse:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      x_ule_inverse:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      x_ule_inverse:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @x_ule_inverse(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_ogt:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_ogt:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ogt:
; FINITE-NEXT: maxsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_ogt(double %x) nounwind {
  %c = fcmp ogt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_olt:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_olt:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_olt:
; FINITE-NEXT: minsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_olt(double %x) nounwind {
  %c = fcmp olt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_ogt_inverse:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_ogt_inverse:
; UNSAFE-NEXT: movsd  {{[^,]*}}, %xmm1
; UNSAFE-NEXT: minsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ogt_inverse:
; FINITE-NEXT: movsd  {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_ogt_inverse(double %x) nounwind {
  %c = fcmp ogt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_olt_inverse:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_olt_inverse:
; UNSAFE-NEXT: movsd  {{[^,]*}}, %xmm1
; UNSAFE-NEXT: maxsd  %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_olt_inverse:
; FINITE-NEXT: movsd  {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_olt_inverse(double %x) nounwind {
  %c = fcmp olt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_oge:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      y_oge:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_oge:
; FINITE-NEXT: maxsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_oge(double %x) nounwind {
  %c = fcmp oge double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_ole:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      y_ole:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ole:
; FINITE-NEXT: minsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_ole(double %x) nounwind {
  %c = fcmp ole double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_oge_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      y_oge_inverse:
; UNSAFE-NEXT: movsd   {{[^,]*}}, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_oge_inverse:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_oge_inverse(double %x) nounwind {
  %c = fcmp oge double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_ole_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      y_ole_inverse:
; UNSAFE-NEXT: movsd   {{[^,]*}}, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ole_inverse:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_ole_inverse(double %x) nounwind {
  %c = fcmp ole double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_ugt:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      y_ugt:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ugt:
; FINITE-NEXT: maxsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_ugt(double %x) nounwind {
  %c = fcmp ugt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_ult:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      y_ult:
; UNSAFE-NEXT: minsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ult:
; FINITE-NEXT: minsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_ult(double %x) nounwind {
  %c = fcmp ult double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_ugt_inverse:
; CHECK:      ucomisd %xmm0, %xmm1
; UNSAFE:      y_ugt_inverse:
; UNSAFE-NEXT: movsd   {{[^,]*}}, %xmm1
; UNSAFE-NEXT: minsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ugt_inverse:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_ugt_inverse(double %x) nounwind {
  %c = fcmp ugt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_ult_inverse:
; CHECK:      ucomisd %xmm1, %xmm0
; UNSAFE:      y_ult_inverse:
; UNSAFE-NEXT: movsd   {{[^,]*}}, %xmm1
; UNSAFE-NEXT: maxsd   %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ult_inverse:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_ult_inverse(double %x) nounwind {
  %c = fcmp ult double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_uge:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_uge:
; UNSAFE-NEXT: maxsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_uge:
; FINITE-NEXT: maxsd  {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_uge(double %x) nounwind {
  %c = fcmp uge double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_ule:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_ule:
; UNSAFE-NEXT: minsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ule:
; FINITE-NEXT: minsd  {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @y_ule(double %x) nounwind {
  %c = fcmp ule double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK:      y_uge_inverse:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_uge_inverse:
; UNSAFE-NEXT: movsd {{[^,]*}}, %xmm1
; UNSAFE-NEXT: minsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_uge_inverse:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_uge_inverse(double %x) nounwind {
  %c = fcmp uge double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK:      y_ule_inverse:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE:      y_ule_inverse:
; UNSAFE-NEXT: movsd {{[^,]*}}, %xmm1
; UNSAFE-NEXT: maxsd %xmm0, %xmm1
; UNSAFE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE:      y_ule_inverse:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @y_ule_inverse(double %x) nounwind {
  %c = fcmp ule double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}
; Test a few more misc. cases.

; CHECK: clampTo3k_a:
; CHECK: minsd
; UNSAFE: clampTo3k_a:
; UNSAFE: minsd
; FINITE: clampTo3k_a:
; FINITE: minsd
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
; FINITE: clampTo3k_b:
; FINITE: minsd
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
; FINITE: clampTo3k_c:
; FINITE: maxsd
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
; FINITE: clampTo3k_d:
; FINITE: maxsd
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
; FINITE: clampTo3k_e:
; FINITE: maxsd
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
; FINITE: clampTo3k_f:
; FINITE: maxsd
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
; FINITE: clampTo3k_g:
; FINITE: minsd
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
; FINITE: clampTo3k_h:
; FINITE: minsd
define double @clampTo3k_h(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; UNSAFE: maxpd:
; UNSAFE: maxpd
define <2 x double> @maxpd(<2 x double> %x, <2 x double> %y) {
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; UNSAFE: minpd:
; UNSAFE: minpd
define <2 x double> @minpd(<2 x double> %x, <2 x double> %y) {
  %min_is_x = fcmp ole <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; UNSAFE: maxps:
; UNSAFE: maxps
define <4 x float> @maxps(<4 x float> %x, <4 x float> %y) {
  %max_is_x = fcmp oge <4 x float> %x, %y
  %max = select <4 x i1> %max_is_x, <4 x float> %x, <4 x float> %y
  ret <4 x float> %max
}

; UNSAFE: minps:
; UNSAFE: minps
define <4 x float> @minps(<4 x float> %x, <4 x float> %y) {
  %min_is_x = fcmp ole <4 x float> %x, %y
  %min = select <4 x i1> %min_is_x, <4 x float> %x, <4 x float> %y
  ret <4 x float> %min
}
