; RUN: llc -O3 -mtriple=x86_64-unknown -mcpu=x86-64 -mattr=+sse2,+pclmul < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -O3 -mtriple=x86_64-unknown -mcpu=x86-64 -mattr=+avx2,+pclmul < %s | FileCheck %s --check-prefix=AVX

declare <2 x i64> @llvm.x86.pclmulqdq(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <2 x i64> @commute_lq_lq(<2 x i64>* %a0, <2 x i64> %a1) #0 {
  ;SSE-LABEL: commute_lq_lq
  ;SSE:       pclmulqdq $0, (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_lq_lq
  ;AVX:       vpclmulqdq $0, (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x i64>, <2 x i64>* %a0
  %2 = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %1, <2 x i64> %a1, i8 0)
  ret <2 x i64> %2
}

define <2 x i64> @commute_lq_hq(<2 x i64>* %a0, <2 x i64> %a1) #0 {
  ;SSE-LABEL: commute_lq_hq
  ;SSE:       pclmulqdq $1, (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_lq_hq
  ;AVX:       vpclmulqdq $1, (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x i64>, <2 x i64>* %a0
  %2 = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %1, <2 x i64> %a1, i8 16)
  ret <2 x i64> %2
}

define <2 x i64> @commute_hq_lq(<2 x i64>* %a0, <2 x i64> %a1) #0 {
  ;SSE-LABEL: commute_hq_lq
  ;SSE:       pclmulqdq $16, (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_hq_lq
  ;AVX:       vpclmulqdq $16, (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x i64>, <2 x i64>* %a0
  %2 = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %1, <2 x i64> %a1, i8 1)
  ret <2 x i64> %2
}

define <2 x i64> @commute_hq_hq(<2 x i64>* %a0, <2 x i64> %a1) #0 {
  ;SSE-LABEL: commute_hq_hq
  ;SSE:       pclmulqdq $17, (%rdi), %xmm0
  ;SSE-NEXT:  retq

  ;AVX-LABEL: commute_hq_hq
  ;AVX:       vpclmulqdq $17, (%rdi), %xmm0, %xmm0
  ;AVX-NEXT:  retq

  %1 = load <2 x i64>, <2 x i64>* %a0
  %2 = call <2 x i64> @llvm.x86.pclmulqdq(<2 x i64> %1, <2 x i64> %a1, i8 17)
  ret <2 x i64> %2
}
