; RUN: llvm-as < %s | llc -fast-isel -march=x86-64 | FileCheck %s

; CHECK: doo:
; CHECK: xorpd
define double @doo(double %x) nounwind {
  %y = fsub double -0.0, %x
  ret double %y
}

; CHECK: foo:
; CHECK: xorps
define float @foo(float %x) nounwind {
  %y = fsub float -0.0, %x
  ret float %y
}
