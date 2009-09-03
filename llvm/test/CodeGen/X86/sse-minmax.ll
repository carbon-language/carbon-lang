; RUN: llvm-as < %s | llc -march=x86-64 | FileCheck %s

; CHECK: clampTo3k_a:
; CHECK: minsd
define double @clampTo3k_a(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_b:
; CHECK: minsd
define double @clampTo3k_b(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_c:
; CHECK: maxsd
define double @clampTo3k_c(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_d:
; CHECK: maxsd
define double @clampTo3k_d(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_e:
; CHECK: maxsd
define double @clampTo3k_e(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_f:
; CHECK: maxsd
define double @clampTo3k_f(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_g:
; CHECK: minsd
define double @clampTo3k_g(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK: clampTo3k_h:
; CHECK: minsd
define double @clampTo3k_h(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}
