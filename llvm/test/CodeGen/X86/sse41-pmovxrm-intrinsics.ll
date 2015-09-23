; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=CHECK --check-prefix=AVX

define <8 x i16> @test_llvm_x86_sse41_pmovsxbw(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxbw
; SSE41: pmovsxbw (%rdi), %xmm0
; AVX:  vpmovsxbw (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %3 = sext <8 x i8> %2 to <8 x i16>
  ret <8 x i16> %3
}

define <4 x i32> @test_llvm_x86_sse41_pmovsxbd(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxbd
; SSE41: pmovsxbd (%rdi), %xmm0
; AVX:  vpmovsxbd (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i8> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @test_llvm_x86_sse41_pmovsxbq(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxbq
; SSE41: pmovsxbq (%rdi), %xmm0
; AVX:  vpmovsxbq (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = shufflevector <16 x i8> %1, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i8> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <4 x i32> @test_llvm_x86_sse41_pmovsxwd(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxwd
; SSE41: pmovsxwd (%rdi), %xmm0
; AVX:  vpmovsxwd (%rdi), %xmm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %3 = sext <4 x i16> %2 to <4 x i32>
  ret <4 x i32> %3
}

define <2 x i64> @test_llvm_x86_sse41_pmovsxwq(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxwq
; SSE41: pmovsxwq (%rdi), %xmm0
; AVX:  vpmovsxwq (%rdi), %xmm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = shufflevector <8 x i16> %1, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i16> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <2 x i64> @test_llvm_x86_sse41_pmovsxdq(<4 x i32>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovsxdq
; SSE41: pmovsxdq (%rdi), %xmm0
; AVX:  vpmovsxdq (%rdi), %xmm0
  %1 = load <4 x i32>, <4 x i32>* %a, align 1
  %2 = shufflevector <4 x i32> %1, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %3 = sext <2 x i32> %2 to <2 x i64>
  ret <2 x i64> %3
}

define <8 x i16> @test_llvm_x86_sse41_pmovzxbw(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxbw
; SSE41: pmovzxbw (%rdi), %xmm0
; AVX:  vpmovzxbw (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = call <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8> %1)
  ret <8 x i16> %2
}

define <4 x i32> @test_llvm_x86_sse41_pmovzxbd(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxbd
; SSE41: pmovzxbd (%rdi), %xmm0
; AVX:  vpmovzxbd (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = call <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8> %1)
  ret <4 x i32> %2
}

define <2 x i64> @test_llvm_x86_sse41_pmovzxbq(<16 x i8>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxbq
; SSE41: pmovzxbq (%rdi), %xmm0
; AVX:  vpmovzxbq (%rdi), %xmm0
  %1 = load <16 x i8>, <16 x i8>* %a, align 1
  %2 = call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> %1)
  ret <2 x i64> %2
}

define <4 x i32> @test_llvm_x86_sse41_pmovzxwd(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxwd
; SSE41: pmovzxwd (%rdi), %xmm0
; AVX:  vpmovzxwd (%rdi), %xmm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = call <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16> %1)
  ret <4 x i32> %2
}

define <2 x i64> @test_llvm_x86_sse41_pmovzxwq(<8 x i16>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxwq
; SSE41: pmovzxwq (%rdi), %xmm0
; AVX:  vpmovzxwq (%rdi), %xmm0
  %1 = load <8 x i16>, <8 x i16>* %a, align 1
  %2 = call <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16> %1)
  ret <2 x i64> %2
}

define <2 x i64> @test_llvm_x86_sse41_pmovzxdq(<4 x i32>* %a) {
; CHECK-LABEL: test_llvm_x86_sse41_pmovzxdq
; SSE41: pmovzxdq (%rdi), %xmm0
; AVX:  vpmovzxdq (%rdi), %xmm0
  %1 = load <4 x i32>, <4 x i32>* %a, align 1
  %2 = call <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32> %1)
  ret <2 x i64> %2
}

declare <2 x i64> @llvm.x86.sse41.pmovzxdq(<4 x i32>)
declare <2 x i64> @llvm.x86.sse41.pmovzxwq(<8 x i16>)
declare <4 x i32> @llvm.x86.sse41.pmovzxwd(<8 x i16>)
declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>)
declare <4 x i32> @llvm.x86.sse41.pmovzxbd(<16 x i8>)
declare <8 x i16> @llvm.x86.sse41.pmovzxbw(<16 x i8>)
