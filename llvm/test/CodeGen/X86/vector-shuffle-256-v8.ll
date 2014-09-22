; RUN: llc < %s -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <8 x float> @shuffle_v8f32_00000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00000010(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000010
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,1,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00000200(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000200
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,2,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00003000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00003000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[3,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00040000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00040000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[2,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00500000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00500000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[1,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[0,2]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_06000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_06000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_70000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_70000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[3,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,2],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_01014545(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_01014545
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,1,0,1,4,5,4,5]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00112233(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00112233
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm0[0,0,1,1]
; ALL-NEXT:    vunpckhps {{.*}} # xmm0 = xmm0[2,2,3,3]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00001111(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00001111
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,1,1,1]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_81a3c5e7(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_81a3c5e7
; ALL:       # BB#0:
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3],ymm1[4],ymm0[5],ymm1[6],ymm0[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08080808(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08080808
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm1[0,0,2,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,0,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08084c4c(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08084c4c
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm0[0,0],ymm1[0,0],ymm0[4,4],ymm1[4,4]
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm0[0,2,1,3,4,6,5,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_8823cc67(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_8823cc67
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[0,0],ymm0[2,3],ymm1[4,4],ymm0[6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_9832dc76(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_9832dc76
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[1,0],ymm0[3,2],ymm1[5,4],ymm0[7,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_9810dc54(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_9810dc54
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[1,0],ymm0[1,0],ymm1[5,4],ymm0[5,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08194c5d(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08194c5d
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*}} # ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_2a3b6e7f(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_2a3b6e7f
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhps {{.*}} # ymm0 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08192a3b(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm1[0,2,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm1[0,0,2,1]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08991abb(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08991abb
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm0[1,1,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm1[0,2,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm2[0],xmm3[1,2,3]
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm1[0,0,1,1]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_091b2d3f(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_091b2d3f
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_09ab1def(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_09ab1def
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[1,1,2,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1,2,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00014445(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00014445
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,0,1,4,4,4,5]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00204464(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00204464
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,0,4,4,6,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_03004744(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_03004744
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,3,0,0,4,7,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10005444(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10005444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_22006644(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_22006644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_33307774(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_33307774
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,3,3,0,7,7,7,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_32107654(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_32107654
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,2,1,0,7,6,5,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00234467(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00234467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00224466(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00224466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,2,4,4,6,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10325476(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10325476
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_11335577(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_11335577
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10235467(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10235467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10225466(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10225466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,2,2,5,4,6,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00015444(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00015444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,0,1,4,4,4,5]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00204644(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00204644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,2,0,0,4,6,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,0,4,4,6,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_03004474(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_03004474
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,0,3,0,4,4,7,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,3,0,0,4,7,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10004444(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10004444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,0,0,0,4,4,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_22006446(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_22006446
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,0,0,2,6,4,4,6]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_33307474(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_33307474
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[3,0,3,0,7,4,7,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,3,3,0,7,7,7,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_32104567(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_32104567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[3,2,1,0,7,6,5,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00236744(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00236744
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,3,0,0,6,7,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00226644(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00226644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,2,4,4,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10324567(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10324567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_11334567(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_11334567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_01235467(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_01235467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_01235466(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_01235466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,2,5,4,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_002u6u44(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_002u6u44
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,1,0,0,6,5,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00uu66uu(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00uu66uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,2,2,3,6,6,6,7]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_103245uu(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_103245uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_1133uu67(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_1133uu67
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_0uu354uu(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_0uu354uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_uuu3uu66(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_uuu3uu66
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,1,2,2,4,5,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <8 x i32> @shuffle_v8i32_00000000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00000000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00000010(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00000010
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,1,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00000200(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00000200
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,2,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00003000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00003000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[3,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00040000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00040000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[2,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00500000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00500000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[1,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[0,2]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_06000000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_06000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_70000000(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_70000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[3,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,2],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_01014545(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_01014545
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[0,0,2,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00112233(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00112233
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm0[0,0,1,1]
; ALL-NEXT:    vunpckhps {{.*}} # xmm0 = xmm0[2,2,3,3]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00001111(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00001111
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,1,1,1]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_81a3c5e7(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_81a3c5e7
; ALL:       # BB#0:
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3],ymm1[4],ymm0[5],ymm1[6],ymm0[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_08080808(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_08080808
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm1[0,0,2,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,0,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_08084c4c(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_08084c4c
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm0[0,0],ymm1[0,0],ymm0[4,4],ymm1[4,4]
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm0[0,2,1,3,4,6,5,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_8823cc67(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_8823cc67
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[0,0],ymm0[2,3],ymm1[4,4],ymm0[6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_9832dc76(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_9832dc76
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[1,0],ymm0[3,2],ymm1[5,4],ymm0[7,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_9810dc54(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_9810dc54
; ALL:       # BB#0:
; ALL-NEXT:    vshufps {{.*}} # ymm0 = ymm1[1,0],ymm0[1,0],ymm1[5,4],ymm0[5,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_08194c5d(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_08194c5d
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*}} # ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[4],ymm1[4],ymm0[5],ymm1[5]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_2a3b6e7f(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_2a3b6e7f
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhps {{.*}} # ymm0 = ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[6],ymm1[6],ymm0[7],ymm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_08192a3b(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm1[0,2,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm1[0,0,2,1]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_08991abb(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_08991abb
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm0[1,1,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm1[0,2,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm2[0],xmm3[1,2,3]
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm1[0,0,1,1]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_091b2d3f(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_091b2d3f
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_09ab1def(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_09ab1def
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[1,1,2,3]
; ALL-NEXT:    vblendps {{.*}} # xmm2 = xmm3[0],xmm2[1,2,3]
; ALL-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00014445(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00014445
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,0,1,4,4,4,5]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00204464(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00204464
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,0,4,4,6,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_03004744(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_03004744
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,3,0,0,4,7,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10005444(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10005444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_22006644(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_22006644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_33307774(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_33307774
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,3,3,0,7,7,7,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_32107654(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_32107654
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,2,1,0,7,6,5,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00234467(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00234467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00224466(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00224466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,2,4,4,6,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10325476(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10325476
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_11335577(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_11335577
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10235467(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10235467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10225466(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10225466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,2,2,5,4,6,6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00015444(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00015444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,0,1,4,4,4,5]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00204644(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00204644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,2,0,0,4,6,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,0,4,4,6,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_03004474(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_03004474
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,0,3,0,4,4,7,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,3,0,0,4,7,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10004444(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10004444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,0,0,0,4,4,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[1,0,0,0,5,4,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_22006446(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_22006446
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,0,0,2,6,4,4,6]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_33307474(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_33307474
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[3,0,3,0,7,4,7,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[3,3,3,0,7,7,7,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_32104567(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_32104567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[3,2,1,0,7,6,5,4]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00236744(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00236744
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,3,0,0,6,7,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00226644(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00226644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,2,0,0,6,6,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,2,4,4,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_10324567(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_10324567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_11334567(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_11334567
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_01235467(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_01235467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_01235466(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_01235466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,2,5,4,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_002u6u44(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_002u6u44
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,1,0,0,6,5,4,4]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_00uu66uu(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_00uu66uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[2,2,2,3,6,6,6,7]
; ALL-NEXT:    vpermilps {{.*}} # ymm0 = ymm0[0,0,2,3,4,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_103245uu(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_103245uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,3,2,5,4,7,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_1133uu67(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_1133uu67
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,1,3,3,5,5,7,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm1[0,1,2,3],ymm0[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_0uu354uu(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_0uu354uu
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[1,0,2,3,5,4,6,7]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x i32> %shuffle
}

define <8 x i32> @shuffle_v8i32_uuu3uu66(<8 x i32> %a, <8 x i32> %b) {
; ALL-LABEL: @shuffle_v8i32_uuu3uu66
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # ymm1 = ymm0[0,1,2,2,4,5,6,6]
; ALL-NEXT:    vblendps {{.*}} # ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x i32> %shuffle
}
