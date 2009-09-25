; RUN: llc < %s -march=x86-64 -asm-verbose=false | FileCheck %s

; Currently, floating-point selects are lowered to CFG triangles.
; This means that one side of the select is always unconditionally
; evaluated, however with MachineSink we can sink the other side so
; that it's conditionally evaluated.

; CHECK: foo:
; CHECK-NEXT: divsd
; CHECK-NEXT: testb $1, %dil
; CHECK-NEXT: jne

define double @foo(double %x, double %y, i1 %c) nounwind {
  %a = fdiv double %x, 3.2
  %b = fdiv double %y, 3.3
  %z = select i1 %c, double %a, double %b
  ret double %z
}
