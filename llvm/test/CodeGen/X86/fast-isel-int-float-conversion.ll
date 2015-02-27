; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=generic -mattr=+sse2 -O0 --fast-isel-abort=1 < %s | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=generic -mattr=+avx -O0 --fast-isel-abort=1 < %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX


define double @int_to_double_rr(i32 %a) {
; ALL-LABEL: int_to_double_rr:
; SSE2: cvtsi2sdl %edi, %xmm0
; AVX: vcvtsi2sdl %edi, %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = sitofp i32 %a to double
  ret double %0
}

define double @int_to_double_rm(i32* %a) {
; ALL-LABEL: int_to_double_rm:
; SSE2: cvtsi2sdl (%rdi), %xmm0
; AVX: vcvtsi2sdl (%rdi), %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = load i32, i32* %a
  %1 = sitofp i32 %0 to double
  ret double %1
}

define float @int_to_float_rr(i32 %a) {
; ALL-LABEL: int_to_float_rr:
; SSE2: cvtsi2ssl %edi, %xmm0
; AVX: vcvtsi2ssl %edi, %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = sitofp i32 %a to float
  ret float %0
}

define float @int_to_float_rm(i32* %a) {
; ALL-LABEL: int_to_float_rm:
; SSE2: cvtsi2ssl (%rdi), %xmm0
; AVX: vcvtsi2ssl (%rdi), %xmm0, %xmm0
; ALL-NEXT: ret
entry:
  %0 = load i32, i32* %a
  %1 = sitofp i32 %0 to float
  ret float %1
}
