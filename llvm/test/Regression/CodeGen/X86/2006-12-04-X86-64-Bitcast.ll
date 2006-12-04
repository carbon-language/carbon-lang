; PR1033
; RUN: llvm-as < %s | llc -march=x86-64
; XFAIL: *
long %p(double %t) {
  %u = bitcast double %t to long
  ret long %u
}

double %q(long %t) {
  %u = bitcast long %t to double
  ret double %u
}

