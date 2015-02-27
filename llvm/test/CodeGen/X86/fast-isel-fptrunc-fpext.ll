; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefix=ALL --check-prefix=AVX
;
; Verify that fast-isel doesn't select legacy SSE instructions on targets that
; feature AVX.
;
; Test cases are obtained from the following code snippet:
; ///
; double single_to_double_rr(float x) {
;   return (double)x;
; }
; float double_to_single_rr(double x) {
;   return (float)x;
; }
; double single_to_double_rm(float *x) {
;   return (double)*x;
; }
; float double_to_single_rm(double *x) {
;   return (float)*x;
; }
; ///

define double @single_to_double_rr(float %x) {
; ALL-LABEL: single_to_double_rr:
; SSE-NOT: vcvtss2sd
; AVX: vcvtss2sd %xmm0, %xmm0, %xmm0
; ALL: ret
entry:
  %conv = fpext float %x to double
  ret double %conv
}

define float @double_to_single_rr(double %x) {
; ALL-LABEL: double_to_single_rr:
; SSE-NOT: vcvtsd2ss
; AVX: vcvtsd2ss %xmm0, %xmm0, %xmm0
; ALL: ret
entry:
  %conv = fptrunc double %x to float
  ret float %conv
}

define double @single_to_double_rm(float* %x) {
; ALL-LABEL: single_to_double_rm:
; SSE: cvtss2sd (%rdi), %xmm0
; AVX: vmovss (%rdi), %xmm0
; AVX-NEXT: vcvtss2sd %xmm0, %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = load float, float* %x, align 4
  %conv = fpext float %0 to double
  ret double %conv
}

define float @double_to_single_rm(double* %x) {
; ALL-LABEL: double_to_single_rm:
; SSE: cvtsd2ss (%rdi), %xmm0
; AVX: vmovsd (%rdi), %xmm0
; AVX-NEXT: vcvtsd2ss %xmm0, %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = load double, double* %x, align 8
  %conv = fptrunc double %0 to float
  ret float %conv
}
