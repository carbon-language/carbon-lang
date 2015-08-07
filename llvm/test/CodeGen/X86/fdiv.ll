; RUN: llc < %s -march=x86-64 -enable-unsafe-fp-math | FileCheck %s

define double @exact(double %x) {
; Exact division by a constant converted to multiplication.
; CHECK-LABEL: exact:
; CHECK:       ## BB#0:
; CHECK-NEXT:    mulsd {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %div = fdiv double %x, 2.0
  ret double %div
}

define double @inexact(double %x) {
; Inexact division by a constant converted to multiplication.
; CHECK-LABEL: inexact:
; CHECK:       ## BB#0:
; CHECK-NEXT:    mulsd {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %div = fdiv double %x, 0x41DFFFFFFFC00000
  ret double %div
}

define double @funky(double %x) {
; No conversion to multiplication if too funky.
; CHECK-LABEL: funky:
; CHECK:       ## BB#0:
; CHECK-NEXT:    xorpd %xmm1, %xmm1
; CHECK-NEXT:    divsd %xmm1, %xmm0
; CHECK-NEXT:    retq
  %div = fdiv double %x, 0.0
  ret double %div
}

define double @denormal1(double %x) {
; Don't generate multiplication by a denormal.
; CHECK-LABEL: denormal1:
; CHECK:       ## BB#0:
; CHECK-NEXT:    divsd {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %div = fdiv double %x, 0x7FD0000000000001
  ret double %div
}

define double @denormal2(double %x) {
; Don't generate multiplication by a denormal.
; CHECK-LABEL: denormal2:
; CHECK:       ## BB#0:
; CHECK-NEXT:    divsd {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %div = fdiv double %x, 0x7FEFFFFFFFFFFFFF
  ret double %div
}

