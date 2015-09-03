; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd | FileCheck %s --check-prefix=AVX512CD

define <16 x i32> @test_ctlz_d(<16 x i32> %a) {
; AVX512CD-LABEL: test_ctlz_d:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    retq
  %res = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a, i1 false)
  ret <16 x i32> %res
}

define <8 x i64> @test_ctlz_q(<8 x i64> %a) {
; AVX512CD-LABEL: test_ctlz_q:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntq %zmm0, %zmm0
; AVX512CD-NEXT:    retq
  %res = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %a, i1 false)
  ret <8 x i64> %res
}

define <16 x i32> @test_ctlz_d_undef(<16 x i32> %a) {
; AVX512CD-LABEL: test_ctlz_d_undef:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512CD-NEXT:    retq
  %res = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a, i1 -1)
  ret <16 x i32> %res
}

define <8 x i64> @test_ctlz_q_undef(<8 x i64> %a) {
; AVX512CD-LABEL: test_ctlz_q_undef:
; AVX512CD:       ## BB#0:
; AVX512CD-NEXT:    vplzcntq %zmm0, %zmm0
; AVX512CD-NEXT:    retq
  %res = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %a, i1 -1)
  ret <8 x i64> %res
}

declare <16 x i32> @llvm.ctlz.v16i32(<16 x i32>, i1) nounwind readonly
declare <8 x i64> @llvm.ctlz.v8i64(<8 x i64>, i1) nounwind readonly
