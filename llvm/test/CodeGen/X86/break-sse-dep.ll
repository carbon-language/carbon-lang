; RUN: llc < %s -march=x86-64 -mattr=+sse2 | FileCheck %s

define double @t1(float* nocapture %x) nounwind readonly ssp {
entry:
; CHECK: t1:
; CHECK: movss (%rdi), %xmm0
; CHECK; cvtss2sd %xmm0, %xmm0

  %0 = load float* %x, align 4
  %1 = fpext float %0 to double
  ret double %1
}

define float @t2(double* nocapture %x) nounwind readonly ssp optsize {
entry:
; CHECK: t2:
; CHECK; cvtsd2ss (%rdi), %xmm0
  %0 = load double* %x, align 8
  %1 = fptrunc double %0 to float
  ret float %1
}
