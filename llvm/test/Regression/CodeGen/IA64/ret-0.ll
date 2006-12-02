; RUN: llvm-upgrade < %s | llvm-as | llc -march=ia64

double %test() {
  ret double 0.0
}
