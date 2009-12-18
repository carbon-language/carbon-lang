; RUN: llc < %s -march=x86-64 -mattr=+sse2,+break-sse-dep | FileCheck %s --check-prefix=YES
; RUN: llc < %s -march=x86-64 -mattr=+sse2,-break-sse-dep | FileCheck %s --check-prefix=NO

define double @t1(float* nocapture %x) nounwind readonly ssp {
entry:
; YES: t1:
; YES: movss (%rdi), %xmm0
; YES; cvtss2sd %xmm0, %xmm0

; NO: t1:
; NO; cvtss2sd (%rdi), %xmm0
  %0 = load float* %x, align 4
  %1 = fpext float %0 to double
  ret double %1
}

define float @t2(double* nocapture %x) nounwind readonly ssp {
entry:
; YES: t2:
; YES: movsd (%rdi), %xmm0
; YES; cvtsd2ss %xmm0, %xmm0

; NO: t2:
; NO; cvtsd2ss (%rdi), %xmm0
  %0 = load double* %x, align 8
  %1 = fptrunc double %0 to float
  ret float %1
}
