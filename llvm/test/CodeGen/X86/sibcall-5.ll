; RUN: llc < %s -march=x86-64 | FileCheck %s

; Sibcall optimization of expanded libcalls.
; rdar://8707777

define double @foo(double %a) nounwind readonly ssp {
entry:
; CHECK: foo:
; CHECK: jmp {{_?}}sin
  %0 = tail call double @sin(double %a) nounwind readonly
  ret double %0
}

define float @bar(float %a) nounwind readonly ssp {
; CHECK: bar:
; CHECK: jmp {{_?}}sinf
entry:
  %0 = tail call float @sinf(float %a) nounwind readonly
  ret float %0
}

declare float @sinf(float) nounwind readonly

declare double @sin(double) nounwind readonly
