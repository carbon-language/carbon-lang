; RUN: llc < %s -march=x86-64 | FileCheck %s
; RUN: llc < %s -march=x86-64 -enable-unsafe-fp-math | FileCheck -check-prefix=UNSAFE %s

define double @exact(double %x) {
; Exact division by a constant always converted to multiplication.
; CHECK: @exact
; CHECK: mulsd
; UNSAFE: @exact
; UNSAFE: mulsd
  %div = fdiv double %x, 2.0
  ret double %div
}

define double @inexact(double %x) {
; Inexact division by a constant converted to multiplication if unsafe-math.
; CHECK: @inexact
; CHECK: divsd
; UNSAFE: @inexact
; UNSAFE: mulsd
  %div = fdiv double %x, 0x41DFFFFFFFC00000 
  ret double %div
}

define double @funky(double %x) {
; No conversion to multiplication if too funky.
; CHECK: @funky
; CHECK: divsd
; UNSAFE: @funky
; UNSAFE: divsd
  %div = fdiv double %x, 0.0
  ret double %div
}
