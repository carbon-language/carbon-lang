; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512bw -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512BW

target triple = "x86_64-unknown-unknown"

define <8 x double> @shuffle_v8f64_00000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00000000
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000010(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00000010
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000200(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00000200
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00003000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00003000
; ALL:       # BB#0:
; ALL-NEXT:    vpermpd {{.*}} # ymm1 = ymm0[0,0,0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00040000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00040000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00500000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00500000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_06000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_06000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_70000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_70000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermpd {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01014545(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_01014545
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00112233(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00112233
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm1[0,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00001111(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00001111
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_81a3c5e7(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_81a3c5e7
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm1[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08080808(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_08080808
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08084c4c(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_08084c4c
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm2
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm3
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm2, %ymm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_8823cc67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_8823cc67
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9832dc76(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_9832dc76
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm3[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9810dc54(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_9810dc54
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm3[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08194c5d(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_08194c5d
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm5 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm5[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm4[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm1[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_2a3b6e7f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_2a3b6e7f
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm5 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm5[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm4[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm1[1,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08192a3b(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm5 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm5[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm4[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm2, %ymm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm1[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08991abb(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_08991abb
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_091b2d3f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_091b2d3f
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm4[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm3[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_09ab1def(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_09ab1def
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm3[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00014445(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00014445
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm1[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204464(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00204464
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_03004744
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm1[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10005444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10005444
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_22006644
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307774(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_33307774
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32107654(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_32107654
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00234467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00234467
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00224466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00224466
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10325476(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10325476
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11335577(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_11335577
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10235467
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10225466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10225466
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00015444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00015444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00204644
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_03004474
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10004444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10004444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006446(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_22006446
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm3 = xmm2[0],xmm1[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm3, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_33307474
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm2[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32104567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_32104567
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00236744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00236744
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00226644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00226644
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10324567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_10324567
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11334567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_11334567
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_01235467
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_01235466
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_002u6u44(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_002u6u44
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm0[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00uu66uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_00uu66uu
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm1
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_103245uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_103245uu
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_1133uu67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_1133uu67
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_0uu354uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_0uu354uu
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm2[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_uuu3uu66(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_uuu3uu66
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vpermpd {{.*}} # ymm1 = ymm1[0,1,2,0]
; ALL-NEXT:    vextractf32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vbroadcastsd %xmm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_c348cda0(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_c348cda0
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextractf32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm4 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm4 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm4, %ymm2
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm4
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm4[0],xmm1[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 12, i32 3, i32 4, i32 8, i32 12, i32 13, i32 10, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_f511235a(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: @shuffle_v8f64_f511235a
; ALL:       # BB#0:
; ALL-NEXT:    vextractf32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vextractf32x4 $2, %zmm0, %xmm3
; ALL-NEXT:    vpermilpd {{.*}} # xmm3 = xmm3[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextractf32x4 $1, %zmm0, %xmm4
; ALL-NEXT:    vpermilpd {{.*}} # xmm5 = xmm4[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm4 = xmm4[0],xmm5[0]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm4, %ymm2
; ALL-NEXT:    vextractf32x4 $3, %zmm1, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm3[0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinsertf64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 15, i32 5, i32 1, i32 1, i32 2, i32 3, i32 5, i32 10>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00000000
; ALL:       # BB#0:
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vpbroadcastq %rax, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000010(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00000010
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000200(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00000200
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00003000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00003000
; ALL:       # BB#0:
; ALL-NEXT:    vpermq {{.*}} # ymm1 = ymm0[0,0,0,0]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00040000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00040000
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00500000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00500000
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_06000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_06000000
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_70000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_70000000
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpermq {{.*}} # ymm0 = ymm0[0,0,0,0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01014545(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_01014545
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00112233(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00112233
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm1[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00001111(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00001111
; ALL:       # BB#0:
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_81a3c5e7(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_81a3c5e7
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm4
; ALL-NEXT:    vpextrq $1, %xmm4, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm4
; ALL-NEXT:    vpextrq $1, %xmm4, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm1[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08080808(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_08080808
; ALL:       # BB#0:
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08084c4c(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_08084c4c
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm2, %ymm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_8823cc67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_8823cc67
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinserti128 $1, %xmm3, %ymm2, %ymm2
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9832dc76(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_9832dc76
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm3[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9810dc54(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_9810dc54
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm3[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08194c5d(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_08194c5d
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm4 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm4, %ymm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm3, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_2a3b6e7f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_2a3b6e7f
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm4 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm4, %ymm2
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm3, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08192a3b(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm4 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm4, %ymm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm3, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08991abb(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_08991abb
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_091b2d3f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_091b2d3f
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm4[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm4
; ALL-NEXT:    vpextrq $1, %xmm4, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_09ab1def(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_09ab1def
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm4[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm4
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm3[0],xmm4[0]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm3, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00014445(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00014445
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm1[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204464(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00204464
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_03004744
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm1[0],xmm2[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10005444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10005444
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_22006644
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307774(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_33307774
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32107654(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_32107654
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00234467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00234467
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00224466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00224466
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10325476(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10325476
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11335577(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_11335577
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10235467
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10225466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10225466
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00015444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00015444
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[0,1,0,1]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00204644
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_03004474
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm3[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10004444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10004444
; ALL:       # BB#0:
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006446(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_22006446
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm3, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_33307474
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm2[0],xmm0[0]
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32104567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_32104567
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00236744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00236744
; ALL:       # BB#0:
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00226644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00226644
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10324567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_10324567
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm3[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11334567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_11334567
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_01235467
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_01235466
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm3[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_002u6u44(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_002u6u44
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00uu66uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_00uu66uu
; ALL:       # BB#0:
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpbroadcastq %xmm1, %ymm1
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_103245uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_103245uu
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm0[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm2[0]
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_1133uu67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_1133uu67
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm1
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm0, %ymm2, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_0uu354uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_0uu354uu
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpshufd {{.*}} # xmm2 = xmm2[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_uuu3uu66(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_uuu3uu66
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpermq {{.*}} # ymm1 = ymm1[0,1,2,0]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpbroadcastq %xmm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_6caa87e5(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: @shuffle_v8i64_6caa87e5
; ALL:       # BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm1, %xmm2
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm3
; ALL-NEXT:    vpextrq $1, %xmm3, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm0
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm3 = xmm1[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm3, %ymm2
; ALL-NEXT:    vextracti32x4 $2, %zmm1, %xmm3
; ALL-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vextracti32x4 $1, %zmm1, %xmm1
; ALL-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,1,0,1]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm2, %zmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 6, i32 12, i32 10, i32 10, i32 8, i32 7, i32 14, i32 5>
  ret <8 x i64> %shuffle
}
