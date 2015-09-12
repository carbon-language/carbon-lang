; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=avx < %s | FileCheck %s --check-prefix=AVX

; Verify that 128-bit vector logical ops are reassociated.

define <4 x i32> @reassociate_and_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_and_v4i32:
; SSE:       # BB#0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    pand %xmm3, %xmm2
; SSE-NEXT:    pand %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_and_v4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = and <4 x i32> %x2, %t0
  %t2 = and <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

define <4 x i32> @reassociate_or_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_or_v4i32:
; SSE:       # BB#0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    por %xmm3, %xmm2
; SSE-NEXT:    por %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_or_v4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = or <4 x i32> %x2, %t0
  %t2 = or <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

define <4 x i32> @reassociate_xor_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_xor_v4i32:
; SSE:       # BB#0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    pxor %xmm3, %xmm2
; SSE-NEXT:    pxor %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_xor_v4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpxor %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = xor <4 x i32> %x2, %t0
  %t2 = xor <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

