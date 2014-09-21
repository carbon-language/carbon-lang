; RUN: llc < %s -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <8 x float> @shuffle_v8f32_00000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00000010(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000010
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,1,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00000200(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00000200
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,2,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00003000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00003000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[3,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00040000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00040000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[2,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00500000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00500000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[1,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm0[0,0],xmm1[0,2]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_06000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_06000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[2,0],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_70000000(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_70000000
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm1
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[3,0],xmm0[0,0]
; ALL-NEXT:    vshufps {{.*}} # xmm1 = xmm1[0,2],xmm0[0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_01014545(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_01014545
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00112233(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00112233
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm0[0,0,1,1]
; ALL-NEXT:    vunpckhps {{.*}} # xmm0 = xmm0[2,2,3,3]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00001111(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00001111
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,0]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,1,1,1]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08192a3b(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm1[0,2,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps  $10, %xmm2, %xmm3, %xmm2 # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm1[0,0,2,1]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps  $10, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128  $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_08991abb(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_08991abb
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm2 = xmm0[1,1,2,3]
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm1[0,2,3,3]
; ALL-NEXT:    vblendps  $1, %xmm2, %xmm3, %xmm2 # xmm2 = xmm2[0],xmm3[1,2,3]
; ALL-NEXT:    vunpcklps {{.*}} # xmm1 = xmm1[0,0,1,1]
; ALL-NEXT:    vblendps  $1, %xmm0, %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128  $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_091b2d3f(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_091b2d3f
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[2,1,3,3]
; ALL-NEXT:    vblendps  $10, %xmm2, %xmm3, %xmm2 # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3]
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,1,1,3]
; ALL-NEXT:    vblendps  $10, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; ALL-NEXT:    vinsertf128  $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_09ab1def(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_09ab1def
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128  $1, %ymm1, %xmm2
; ALL-NEXT:    vpermilps {{.*}} # xmm3 = xmm0[1,1,2,3]
; ALL-NEXT:    vblendps  $1, %xmm3, %xmm2, %xmm2 # xmm2 = xmm3[0],xmm2[1,2,3]
; ALL-NEXT:    vblendps  $1, %xmm0, %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[1,2,3]
; ALL-NEXT:    vinsertf128  $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00014445(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00014445
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,0,1]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,0,1]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00204464(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00204464
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,2,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,2,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_03004744(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_03004744
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,3,0,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,3,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10005444(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10005444
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[1,0,0,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,0,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_22006644(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_22006644
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[2,2,0,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[2,2,0,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_33307774(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_33307774
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[3,3,3,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[3,3,3,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_32107654(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_32107654
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[3,2,1,0]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[3,2,1,0]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00234467(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00234467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[0,0,2,3]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[0,0,2,3]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_00224466(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_00224466
; ALL:       # BB#0:
; ALL-NEXT:    vmovsldup {{.*}} # xmm1 = xmm0[0,0,2,2]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vmovsldup {{.*}} # xmm0 = xmm0[0,0,2,2]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10325476(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10325476
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[1,0,3,2]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,0,3,2]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_11335577(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_11335577
; ALL:       # BB#0:
; ALL-NEXT:    vmovshdup {{.*}} # xmm1 = xmm0[1,1,3,3]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vmovshdup {{.*}} # xmm0 = xmm0[1,1,3,3]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10235467(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10235467
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[1,0,2,3]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,0,2,3]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x float> %shuffle
}

define <8 x float> @shuffle_v8f32_10225466(<8 x float> %a, <8 x float> %b) {
; ALL-LABEL: @shuffle_v8f32_10225466
; ALL:       # BB#0:
; ALL-NEXT:    vpermilps {{.*}} # xmm1 = xmm0[1,0,2,2]
; ALL-NEXT:    vextractf128  $1, %ymm0, %xmm0
; ALL-NEXT:    vpermilps {{.*}} # xmm0 = xmm0[1,0,2,2]
; ALL-NEXT:    vinsertf128  $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x float> %shuffle
}
