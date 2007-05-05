; RUN: llvm-upgrade < %s | llvm-as | llc
; XFAIL: ia64


declare bool %llvm.isunordered.f64(double, double)

bool %test(double %X, double %Y) {
  %tmp27 = call bool %llvm.isunordered.f64( double %X, double %Y)
  ret bool %tmp27
}
