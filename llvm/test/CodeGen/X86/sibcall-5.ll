; RUN: llc < %s -mtriple=i386-apple-darwin8 | FileCheck %s --check-prefix=X32
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s --check-prefix=X64

; Sibcall optimization of expanded libcalls.
; rdar://8707777

define double @foo(double %a) nounwind readonly ssp {
entry:
; X32: foo:
; X32: jmp _sin$stub

; X64: foo:
; X64: jmp _sin
  %0 = tail call double @sin(double %a) nounwind readonly
  ret double %0
}

define float @bar(float %a) nounwind readonly ssp {
; X32: bar:
; X32: jmp _sinf$stub

; X64: bar:
; X64: jmp _sinf
entry:
  %0 = tail call float @sinf(float %a) nounwind readonly
  ret float %0
}

declare float @sinf(float) nounwind readonly

declare double @sin(double) nounwind readonly
