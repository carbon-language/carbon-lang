; RUN: llc < %s -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,5,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm0[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,5,4]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,1,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,1,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,3,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,3,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,3,0,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,3,0,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm2[0],xmm1[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,1,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,1,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,3,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,3,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,3,0,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,3,0,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI16_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI16_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI17_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI17_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[3,3,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,7,7,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[3,3,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,7,7,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[3,3,3,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,7,7,7,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[3,3,3,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,7,7,7]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 3, i32 3, i32 3, i32 3, i32 7, i32 7, i32 7, i32 7, i32 11, i32 11, i32 11, i32 11, i32 15, i32 15, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,6,6]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[1,1,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,5,5,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[1,1,3,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,5,7,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[1,1,3,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,5,5,7,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[1,1,3,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,5,7,7]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_00_00_00_00_00_00_01_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_00_00_00_00_00_02_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_00_00_00_00_03_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_00_00_00_04_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_00_00_05_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_00_06_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3],xmm1[4],xmm0[5],xmm1[6],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_16_01_18_03_20_05_22_07_24_09_26_11_28_13_30_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm3
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3],xmm1[4],xmm0[5],xmm1[6],xmm0[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 1, i32 18, i32 3, i32 20, i32 5, i32 22, i32 7, i32 24, i32 9, i32 26, i32 11, i32 28, i32 13, i32 30, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm1[0,1,0,1,4,5,0,1,0,1,0,1,12,13,0,1]
; AVX1-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,0,0,0]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_16_00_16_00_16_00_16_00_16_00_16_00_16_00_16
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm1[0,1,0,1,4,5,0,1,0,1,0,1,12,13,0,1]
; AVX2-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,0,0,0]
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vmovdqa .LCPI31_0(%rip), %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufd {{.*}} # xmm4 = xmm4[0,0,0,0]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm4[0],xmm2[1],xmm4[2],xmm2[3],xmm4[4],xmm2[5],xmm4[6],xmm2[7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,0,0,0]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_16_00_16_00_16_00_16_08_24_08_24_08_24_08_24
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vmovdqa .LCPI31_0(%rip), %xmm3
; AVX2-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm4
; AVX2-NEXT:    vpshufd {{.*}} # xmm4 = xmm4[0,0,0,0]
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm4[0],xmm2[1],xmm4[2],xmm2[3],xmm4[4],xmm2[5],xmm4[6],xmm2[7]
; AVX2-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX2-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,0,0,0]
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 0, i32 16, i32 8, i32 24, i32 8, i32 24, i32 8, i32 24, i32 8, i32 24>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshuflw {{.*}} # xmm3 = xmm3[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_16_16_16_16_04_05_06_07_24_24_24_24_12_13_14_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm3
; AVX2-NEXT:    vpshuflw {{.*}} # xmm3 = xmm3[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpblendd $-16, %xmm2, %xmm3, %xmm2
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpblendd $-16, %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 4, i32 5, i32 6, i32 7, i32 24, i32 24, i32 24, i32 24, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufhw {{.*}} # xmm2 = xmm2[0,1,2,3,7,6,5,4]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpshuflw {{.*}} # xmm3 = xmm3[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_19_18_17_16_07_06_05_04_27_26_25_24_15_14_13_12
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm2
; AVX2-NEXT:    vpshufhw {{.*}} # xmm2 = xmm2[0,1,2,3,7,6,5,4]
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm3
; AVX2-NEXT:    vpshuflw {{.*}} # xmm3 = xmm3[3,2,1,0,4,5,6,7]
; AVX2-NEXT:    vpblendd $-16, %xmm2, %xmm3, %xmm2
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,2,1,0,4,5,6,7]
; AVX2-NEXT:    vpblendd $-16, %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 19, i32 18, i32 17, i32 16, i32 7, i32 6, i32 5, i32 4, i32 27, i32 26, i32 25, i32 24, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpshuflw {{.*}} # xmm2 = xmm2[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpshufd {{.*}} # xmm3 = xmm3[0,1,0,1]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm3 = xmm3[0,1,2,3,7,6,5,4]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm2[0,1,2,3],xmm3[4,5,6,7]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,2,1,0,4,5,6,7]
; AVX1-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_19_18_17_16_03_02_01_00_27_26_25_24_11_10_09_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpshuflw {{.*}} # xmm2 = xmm2[3,2,1,0,4,5,6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpshufd {{.*}} # xmm3 = xmm3[0,1,0,1]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm3 = xmm3[0,1,2,3,7,6,5,4]
; AVX2-NEXT:    vpblendd $-16, %xmm3, %xmm2, %xmm2
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[3,2,1,0,4,5,6,7]
; AVX2-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX2-NEXT:    vpblendd $-16, %xmm0, %xmm1, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 19, i32 18, i32 17, i32 16, i32 3, i32 2, i32 1, i32 0, i32 27, i32 26, i32 25, i32 24, i32 11, i32 10, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI35_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_08_08_08_08_08_08_09_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI35_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 9, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI36_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_08_08_08_10_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI36_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI37_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_08_11_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI37_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 11, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI38_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_12_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI38_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 12, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI39_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_13_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI39_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 13, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI40_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_08_14_08_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI40_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 14, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa .LCPI41_0(%rip), %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_15_08_08_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vmovdqa .LCPI41_0(%rip), %xmm2
; AVX2-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 15, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm2 = xmm2[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpmovzxwd %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm1[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vpmovzxwd %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_16_01_17_02_18_03_19_08_24_09_25_10_26_11_27
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm2 = xmm2[0,0,1,1,2,2,3,3]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpmovzxwd %xmm3, %xmm3
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm1[0,0,1,1,2,2,3,3]
; AVX2-NEXT:    vpmovzxwd %xmm0, %xmm0
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm2 = xmm2[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm3 = xmm3[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm1 = xmm1[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_04_20_05_21_06_22_07_23_12_28_13_29_14_30_15_31
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm2 = xmm2[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm3 = xmm3[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm1 = xmm1[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm2 = xmm2[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm3 = xmm3[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm1[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vpmovzxwd %xmm0, %xmm0
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_16_01_17_02_18_03_19_12_28_13_29_14_30_15_31
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm2 = xmm2[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm3 = xmm3[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm1 = xmm1[0,0,1,1,2,2,3,3]
; AVX2-NEXT:    vpmovzxwd %xmm0, %xmm0
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpunpcklwd {{.*}} # xmm2 = xmm2[0,0,1,1,2,2,3,3]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpmovzxwd %xmm3, %xmm3
; AVX1-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm1 = xmm1[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpunpckhwd {{.*}} # xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_04_20_05_21_06_22_07_23_08_24_09_25_10_26_11_27
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm1, %xmm2
; AVX2-NEXT:    vpunpcklwd {{.*}} # xmm2 = xmm2[0,0,1,1,2,2,3,3]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm3
; AVX2-NEXT:    vpmovzxwd %xmm3, %xmm3
; AVX2-NEXT:    vpblendw {{.*}} # xmm2 = xmm3[0],xmm2[1],xmm3[2],xmm2[3],xmm3[4],xmm2[5],xmm3[6],xmm2[7]
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm1 = xmm1[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpunpckhwd {{.*}} # xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX2-NEXT:    vpblendw {{.*}} # xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3],xmm0[4],xmm1[5],xmm0[6],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_00_01_00_08_09_08_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,2,3,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 8, i32 9, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_00_02_00_00_08_08_10_08_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,0,1,4,5,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 8, i32 8, i32 10, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,6,7,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_00_03_00_00_00_08_08_08_11_08_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,0,1,6,7,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,6,7,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 11, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,8,9,0,1,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_00_04_00_00_00_00_08_08_08_08_12_08_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,0,1,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,8,9,0,1,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 12, i32 8, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,10,11,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_05_00_00_00_00_00_08_08_08_08_08_13_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,0,1,10,11,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,10,11,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 13, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,12,13,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_06_00_00_00_00_00_00_08_08_08_08_08_08_14_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,12,13,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,12,13,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 14, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_07_00_00_00_00_00_00_00_08_08_08_08_08_08_08_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[14,15,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,14,15]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[12,13,12,13,8,9,8,9,4,5,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_00_02_02_04_04_06_06_14_14_12_12_10_10_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,0,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[12,13,12,13,8,9,8,9,4,5,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 14, i32 14, i32 12, i32 12, i32 10, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_04_04_04_04_00_00_00_00_08_08_08_08_12_12_12_12
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,2,3,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,12,13,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_uu_uu_00_00_00_00_00_08_08_uu_uu_08_08_14_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[0,1,2,3,4,5,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,12,13,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 undef, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 undef, i32 undef, i32 8, i32 8, i32 14, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[14,15,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,0,1,14,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_07_uu_00_00_00_00_00_00_08_08_uu_uu_08_08_08_15
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*}} # xmm1 = xmm0[14,15,2,3,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[0,1,0,1,4,5,6,7,0,1,0,1,0,1,14,15]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 7, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 undef, i32 undef, i32 8, i32 8, i32 8, i32 15>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,1,2,2,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[12,13,12,13,12,13,8,9,4,5,4,5,0,1,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_00_uu_uu_02_04_04_uu_06_14_14_uu_12_10_10_08_08
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm0[0,1,2,2,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,4,6,6]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*}} # xmm0 = xmm0[12,13,12,13,12,13,8,9,4,5,4,5,0,1,0,1]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 undef, i32 undef, i32 2, i32 4, i32 4, i32 undef, i32 6, i32 14, i32 14, i32 undef, i32 12, i32 10, i32 10, i32 8, i32 8>
  ret <16 x i16> %shuffle
}

define <16 x i16> @shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12(<16 x i16> %a, <16 x i16> %b) {
; AVX1-LABEL: @shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[2,1,2,3]
; AVX1-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,3,4,5,6,7]
; AVX1-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: @shuffle_v16i16_04_04_04_04_uu_uu_uu_uu_08_08_08_uu_uu_12_12_12
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[2,1,2,3]
; AVX2-NEXT:    vpshuflw {{.*}} # xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm0
; AVX2-NEXT:    vpshuflw {{.*}} # xmm0 = xmm0[0,0,0,3,4,5,6,7]
; AVX2-NEXT:    vpshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 8, i32 8, i32 undef, i32 undef, i32 12, i32 12, i32 12>
  ret <16 x i16> %shuffle
}
