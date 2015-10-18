; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512CD
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512bw | FileCheck %s --check-prefix=AVX512BW

define <8 x i64> @testv8i64(<8 x i64> %in) nounwind {
; ALL-LABEL: testv8i64:
; ALL:       ## BB#0:
; ALL-NEXT:    vplzcntq %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %in, i1 0)
  ret <8 x i64> %out
}

define <8 x i64> @testv8i64u(<8 x i64> %in) nounwind {
; ALL-LABEL: testv8i64u:
; ALL:       ## BB#0:
; ALL-NEXT:    vplzcntq %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <8 x i64> @llvm.ctlz.v8i64(<8 x i64> %in, i1 -1)
  ret <8 x i64> %out
}

define <16 x i32> @testv16i32(<16 x i32> %in) nounwind {
; ALL-LABEL: testv16i32:
; ALL:       ## BB#0:
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %in, i1 0)
  ret <16 x i32> %out
}

define <16 x i32> @testv16i32u(<16 x i32> %in) nounwind {
; ALL-LABEL: testv16i32u:
; ALL:       ## BB#0:
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %in, i1 -1)
  ret <16 x i32> %out
}

define <32 x i16> @testv32i16(<32 x i16> %in) nounwind {
; ALL-LABEL: testv32i16:
; ALL:       ## BB#0:
; ALL-NEXT:    vpmovzxwd %ymm0, %zmm0
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    vpmovdw %zmm0, %ymm0
; ALL-NEXT:    vmovdqa {{.*#+}} ymm2 = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; ALL-NEXT:    vpsubw %ymm2, %ymm0, %ymm0
; ALL-NEXT:    vpmovzxwd %ymm1, %zmm1
; ALL-NEXT:    vplzcntd %zmm1, %zmm1
; ALL-NEXT:    vpmovdw %zmm1, %ymm1
; ALL-NEXT:    vpsubw %ymm2, %ymm1, %ymm1
; ALL-NEXT:    retq
;
; AVX512BW-LABEL: testv32i16:
; AVX512BW:       ## BB#0:
; AVX512BW-NEXT:    vextracti64x4 $1, %zmm0, %ymm1
; AVX512BW-NEXT:    vpmovzxwd %ymm1, %zmm1
; AVX512BW-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512BW-NEXT:    vpmovdw %zmm1, %ymm1
; AVX512BW-NEXT:    vmovdqa {{.*#+}} ymm2 = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX512BW-NEXT:    vpsubw %ymm2, %ymm1, %ymm1
; AVX512BW-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512BW-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512BW-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512BW-NEXT:    vpsubw %ymm2, %ymm0, %ymm0
; AVX512BW-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; AVX512BW-NEXT:    retq
  %out = call <32 x i16> @llvm.ctlz.v32i16(<32 x i16> %in, i1 0)
  ret <32 x i16> %out
}

define <32 x i16> @testv32i16u(<32 x i16> %in) nounwind {
; ALL-LABEL: testv32i16u:
; ALL:       ## BB#0:
; ALL-NEXT:    vpmovzxwd %ymm0, %zmm0
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    vpmovdw %zmm0, %ymm0
; ALL-NEXT:    vmovdqa {{.*#+}} ymm2 = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; ALL-NEXT:    vpsubw %ymm2, %ymm0, %ymm0
; ALL-NEXT:    vpmovzxwd %ymm1, %zmm1
; ALL-NEXT:    vplzcntd %zmm1, %zmm1
; ALL-NEXT:    vpmovdw %zmm1, %ymm1
; ALL-NEXT:    vpsubw %ymm2, %ymm1, %ymm1
; ALL-NEXT:    retq
;
; AVX512BW-LABEL: testv32i16u:
; AVX512BW:       ## BB#0:
; AVX512BW-NEXT:    vextracti64x4 $1, %zmm0, %ymm1
; AVX512BW-NEXT:    vpmovzxwd %ymm1, %zmm1
; AVX512BW-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512BW-NEXT:    vpmovdw %zmm1, %ymm1
; AVX512BW-NEXT:    vmovdqa {{.*#+}} ymm2 = [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX512BW-NEXT:    vpsubw %ymm2, %ymm1, %ymm1
; AVX512BW-NEXT:    vpmovzxwd %ymm0, %zmm0
; AVX512BW-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512BW-NEXT:    vpmovdw %zmm0, %ymm0
; AVX512BW-NEXT:    vpsubw %ymm2, %ymm0, %ymm0
; AVX512BW-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; AVX512BW-NEXT:    retq
  %out = call <32 x i16> @llvm.ctlz.v32i16(<32 x i16> %in, i1 -1)
  ret <32 x i16> %out
}

define <64 x i8> @testv64i8(<64 x i8> %in) nounwind {
; ALL-LABEL: testv64i8:
; ALL:       ## BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm2, %zmm2
; ALL-NEXT:    vplzcntd %zmm2, %zmm2
; ALL-NEXT:    vpmovdb %zmm2, %xmm2
; ALL-NEXT:    vmovdqa {{.*#+}} xmm3 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; ALL-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm0, %zmm0
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    vpmovdb %zmm0, %xmm0
; ALL-NEXT:    vpsubb %xmm3, %xmm0, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm2, %zmm2
; ALL-NEXT:    vplzcntd %zmm2, %zmm2
; ALL-NEXT:    vpmovdb %zmm2, %xmm2
; ALL-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm1, %zmm1
; ALL-NEXT:    vplzcntd %zmm1, %zmm1
; ALL-NEXT:    vpmovdb %zmm1, %xmm1
; ALL-NEXT:    vpsubb %xmm3, %xmm1, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
;
; AVX512BW-LABEL: testv64i8:
; AVX512BW:       ## BB#0:
; AVX512BW-NEXT:    vextracti64x4 $1, %zmm0, %ymm1
; AVX512BW-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm2, %zmm2
; AVX512BW-NEXT:    vplzcntd %zmm2, %zmm2
; AVX512BW-NEXT:    vpmovdb %zmm2, %xmm2
; AVX512BW-NEXT:    vmovdqa {{.*#+}} xmm3 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512BW-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512BW-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm1, %xmm1
; AVX512BW-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; AVX512BW-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm2, %zmm2
; AVX512BW-NEXT:    vplzcntd %zmm2, %zmm2
; AVX512BW-NEXT:    vpmovdb %zmm2, %xmm2
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512BW-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512BW-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm0, %xmm0
; AVX512BW-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX512BW-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; AVX512BW-NEXT:    retq
  %out = call <64 x i8> @llvm.ctlz.v64i8(<64 x i8> %in, i1 0)
  ret <64 x i8> %out
}

define <64 x i8> @testv64i8u(<64 x i8> %in) nounwind {
; ALL-LABEL: testv64i8u:
; ALL:       ## BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm2, %zmm2
; ALL-NEXT:    vplzcntd %zmm2, %zmm2
; ALL-NEXT:    vpmovdb %zmm2, %xmm2
; ALL-NEXT:    vmovdqa {{.*#+}} xmm3 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; ALL-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm0, %zmm0
; ALL-NEXT:    vplzcntd %zmm0, %zmm0
; ALL-NEXT:    vpmovdb %zmm0, %xmm0
; ALL-NEXT:    vpsubb %xmm3, %xmm0, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm2, %zmm2
; ALL-NEXT:    vplzcntd %zmm2, %zmm2
; ALL-NEXT:    vpmovdb %zmm2, %xmm2
; ALL-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; ALL-NEXT:    vpmovzxbd %xmm1, %zmm1
; ALL-NEXT:    vplzcntd %zmm1, %zmm1
; ALL-NEXT:    vpmovdb %zmm1, %xmm1
; ALL-NEXT:    vpsubb %xmm3, %xmm1, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
;
; AVX512BW-LABEL: testv64i8u:
; AVX512BW:       ## BB#0:
; AVX512BW-NEXT:    vextracti64x4 $1, %zmm0, %ymm1
; AVX512BW-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm2, %zmm2
; AVX512BW-NEXT:    vplzcntd %zmm2, %zmm2
; AVX512BW-NEXT:    vpmovdb %zmm2, %xmm2
; AVX512BW-NEXT:    vmovdqa {{.*#+}} xmm3 = [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm1, %zmm1
; AVX512BW-NEXT:    vplzcntd %zmm1, %zmm1
; AVX512BW-NEXT:    vpmovdb %zmm1, %xmm1
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm1, %xmm1
; AVX512BW-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; AVX512BW-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm2, %zmm2
; AVX512BW-NEXT:    vplzcntd %zmm2, %zmm2
; AVX512BW-NEXT:    vpmovdb %zmm2, %xmm2
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm2, %xmm2
; AVX512BW-NEXT:    vpmovzxbd %xmm0, %zmm0
; AVX512BW-NEXT:    vplzcntd %zmm0, %zmm0
; AVX512BW-NEXT:    vpmovdb %zmm0, %xmm0
; AVX512BW-NEXT:    vpsubb %xmm3, %xmm0, %xmm0
; AVX512BW-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX512BW-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; AVX512BW-NEXT:    retq
  %out = call <64 x i8> @llvm.ctlz.v64i8(<64 x i8> %in, i1 -1)
  ret <64 x i8> %out
}

declare <8 x i64> @llvm.ctlz.v8i64(<8 x i64>, i1)
declare <16 x i32> @llvm.ctlz.v16i32(<16 x i32>, i1)
declare <32 x i16> @llvm.ctlz.v32i16(<32 x i16>, i1)
declare <64 x i8> @llvm.ctlz.v64i8(<64 x i8>, i1)
