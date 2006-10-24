; RUN: llvm-as < %s | llc -march=ia64

double %test() {
  ret double 0.0
}
