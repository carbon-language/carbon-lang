; RUN: llvm-as < %s | llc
; XFAIL: ia64|sparcv8


declare bool %llvm.isunordered(double, double)

bool %test(double %X, double %Y) {
  %tmp27 = call bool %llvm.isunordered( double %X, double %Y)
  ret bool %tmp27
}
