; RUN: llvm-as < %s | llc -fast-isel -fast-isel-abort -march=x86-64 | FileCheck %s

; CHECK: doo:
; CHECK: xor
define double @doo(double %x) nounwind {
  %y = fsub double -0.0, %x
  ret double %y
}

; CHECK: foo:
; CHECK: xor
define float @foo(float %x) nounwind {
  %y = fsub float -0.0, %x
  ret float %y
}
