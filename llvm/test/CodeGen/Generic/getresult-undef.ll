; RUN: llc < %s

define double @foo() {
  %t = getresult {double, double} undef, 1
  ret double %t
}
