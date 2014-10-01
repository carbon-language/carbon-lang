; RUN: llc < %s -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_03_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_04_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_05_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_06_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_09_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 9, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_10_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,11,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_11_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,11,0,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_12_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,13,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_13_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,13,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    movl $15, %eax
; AVX1-NEXT:    vmovd %eax, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    movl $15, %eax
; AVX2-NEXT:    vmovd %eax, %xmm1
; AVX2-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX2-NEXT:    vpbroadcastb %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[0]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],zero
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpshufb %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_17_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_17_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[1],zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0],zero,xmm0[0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_17_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,u,u,u,u,u,1,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 17, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_18_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_18_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[2],zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0],zero,xmm0[0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_18_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,u,u,u,u,2,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 18, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_19_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_19_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[3],zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0],zero,xmm0[0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_19_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,u,u,u,3,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 19, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_20_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_20_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[4],zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0],zero,xmm0[0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_20_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,u,u,4,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_21_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_21_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[5],zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0],zero,xmm0[0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_21_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,u,5,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 21, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_22_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_22_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,zero,xmm2[6],zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0],zero,xmm0[0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_22_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,u,6,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 22, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_23_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_23_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,zero,xmm2[7],zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0],zero,xmm0[0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_23_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,u,7,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 23, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_24_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_24_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,zero,xmm2[8],zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0],zero,xmm0[0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_24_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,u,8,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 24, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_25_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_25_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,zero,xmm2[9],zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0],zero,xmm0[0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_25_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,u,9,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 25, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_26_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_26_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,zero,xmm2[10],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0],zero,xmm0[0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_26_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,u,10,u,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 26, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_27_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_27_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,zero,xmm2[11],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0],zero,xmm0[0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_27_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,u,11,u,u,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 27, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_28_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_28_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,zero,xmm2[12],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0],zero,xmm0[0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_28_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,u,12,u,u,u,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 28, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_29_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_29_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,xmm2[13],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0],zero,xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_29_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,u,13,u,u,u,u,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 29, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_30_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_30_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,xmm2[14],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0],zero,xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpor %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_30_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,14,u,u,u,u,u,u,u,u,u,u,u,u,u,u,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_31_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_31_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    movl $128, %eax
; AVX1-NEXT:    vmovd %eax, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[15],zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
; AVX1-NEXT:    vpor %xmm0, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_31_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm1 = ymm0[2,3,0,1]
; AVX2-NEXT:    movl $15, %eax
; AVX2-NEXT:    vmovd %eax, %xmm2
; AVX2-NEXT:    vpxor %ymm3, %ymm3, %ymm3
; AVX2-NEXT:    vinserti128 $0, %xmm2, %ymm3, %ymm2
; AVX2-NEXT:    vpshufb %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpbroadcastb %xmm0, %ymm0
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm1, %ymm1, %ymm1
; AVX2-NEXT:    vpshufb %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,24,24,24,24,24,24,24,24]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15_23_23_23_23_23_23_23_23_31_31_31_31_31_31_31_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15_23_23_23_23_23_23_23_23_31_31_31_31_31_31_31_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [7,7,7,7,7,7,7,7,15,15,15,15,15,15,15,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_07_07_07_07_07_07_07_07_15_15_15_15_15_15_15_15_23_23_23_23_23_23_23_23_31_31_31_31_31_31_31_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[7,7,7,7,7,7,7,7,15,15,15,15,15,15,15,15,23,23,23,23,23,23,23,23,31,31,31,31,31,31,31,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20_24_24_24_24_28_28_28_28(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20_24_24_24_24_28_28_28_28:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_16_16_16_16_20_20_20_20_24_24_24_24_28_28_28_28:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,16,16,16,16,20,20,20,20,24,24,24,24,28,28,28,28]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12, i32 16, i32 16, i32 16, i32 16, i32 20, i32 20, i32 20, i32 20, i32 24, i32 24, i32 24, i32 24, i32 28, i32 28, i32 28, i32 28>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15_19_19_19_19_23_23_23_23_27_27_27_27_31_31_31_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15_19_19_19_19_23_23_23_23_27_27_27_27_31_31_31_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_03_03_03_03_07_07_07_07_11_11_11_11_15_15_15_15_19_19_19_19_23_23_23_23_27_27_27_27_31_31_31_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15,19,19,19,19,23,23,23,23,27,27,27,27,31,31,31,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 3, i32 3, i32 3, i32 3, i32 7, i32 7, i32 7, i32 7, i32 11, i32 11, i32 11, i32 11, i32 15, i32 15, i32 15, i32 15, i32 19, i32 19, i32 19, i32 19, i32 23, i32 23, i32 23, i32 23, i32 27, i32 27, i32 27, i32 27, i32 31, i32 31, i32 31, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14_16_16_18_18_20_20_22_22_24_24_26_26_28_28_30_30(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14_16_16_18_18_20_20_22_22_24_24_26_26_28_28_30_30:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_02_02_04_04_06_06_08_08_10_10_12_12_14_14_16_16_18_18_20_20_22_22_24_24_26_26_28_28_30_30:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14,16,16,18,18,20,20,22,22,24,24,26,26,28,28,30,30]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6, i32 8, i32 8, i32 10, i32 10, i32 12, i32 12, i32 14, i32 14, i32 16, i32 16, i32 18, i32 18, i32 20, i32 20, i32 22, i32 22, i32 24, i32 24, i32 26, i32 26, i32 28, i32 28, i32 30, i32 30>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15_17_17_19_19_21_21_23_23_25_25_27_27_29_29_31_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15_17_17_19_19_21_21_23_23_25_25_27_27_29_29_31_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_01_01_03_03_05_05_07_07_09_09_11_11_13_13_15_15_17_17_19_19_21_21_23_23_25_25_27_27_29_29_31_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15,17,17,19,19,21,21,23,23,25,25,27,27,29,29,31,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7, i32 9, i32 9, i32 11, i32 11, i32 13, i32 13, i32 15, i32 15, i32 17, i32 17, i32 19, i32 19, i32 21, i32 21, i32 23, i32 23, i32 25, i32 25, i32 27, i32 27, i32 29, i32 29, i32 31, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX1:       # BB#0:
; AVX1-NEXT:    movl $15, %eax
; AVX1-NEXT:    vmovd %eax, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00:
; AVX2:       # BB#0:
; AVX2-NEXT:    movl $15, %eax
; AVX2-NEXT:    vmovd %eax, %xmm1
; AVX2-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_33_02_35_04_37_06_39_08_41_10_43_12_45_14_47_16_49_18_51_20_53_22_55_24_57_26_59_28_61_30_63(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_33_02_35_04_37_06_39_08_41_10_43_12_45_14_47_16_49_18_51_20_53_22_55_24_57_26_59_28_61_30_63:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <1,3,5,7,9,11,13,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = <0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm4
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm4[0],xmm2[0],xmm4[1],xmm2[1],xmm4[2],xmm2[2],xmm4[3],xmm2[3],xmm4[4],xmm2[4],xmm4[5],xmm2[5],xmm4[6],xmm2[6],xmm4[7],xmm2[7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm5, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_33_02_35_04_37_06_39_08_41_10_43_12_45_14_47_16_49_18_51_20_53_22_55_24_57_26_59_28_61_30_63:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 33, i32 2, i32 35, i32 4, i32 37, i32 6, i32 39, i32 8, i32 41, i32 10, i32 43, i32 12, i32 45, i32 14, i32 47, i32 16, i32 49, i32 18, i32 51, i32 20, i32 53, i32 22, i32 55, i32 24, i32 57, i32 26, i32 59, i32 28, i32 61, i32 30, i32 63>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_32_01_34_03_36_05_38_07_40_09_42_11_44_13_46_15_48_17_50_19_52_21_54_23_56_25_58_27_60_29_62_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_32_01_34_03_36_05_38_07_40_09_42_11_44_13_46_15_48_17_50_19_52_21_54_23_56_25_58_27_60_29_62_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <1,3,5,7,9,11,13,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = <0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm4
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm4[0],xmm2[0],xmm4[1],xmm2[1],xmm4[2],xmm2[2],xmm4[3],xmm2[3],xmm4[4],xmm2[4],xmm4[5],xmm2[5],xmm4[6],xmm2[6],xmm4[7],xmm2[7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_32_01_34_03_36_05_38_07_40_09_42_11_44_13_46_15_48_17_50_19_52_21_54_23_56_25_58_27_60_29_62_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 32, i32 1, i32 34, i32 3, i32 36, i32 5, i32 38, i32 7, i32 40, i32 9, i32 42, i32 11, i32 44, i32 13, i32 46, i32 15, i32 48, i32 17, i32 50, i32 19, i32 52, i32 21, i32 54, i32 23, i32 56, i32 25, i32 58, i32 27, i32 60, i32 29, i32 62, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastb %xmm1, %xmm1
; AVX2-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[1,3,5,7,9,11,13,15,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpbroadcastw %xmm0, %xmm0
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX2-NEXT:    vinserti128 $1, %xmm0, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_16_48_16_48_16_48_16_48_16_48_16_48_16_48_16_48(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_16_48_16_48_16_48_16_48_16_48_16_48_16_48_16_48:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vpxor %xmm3, %xmm3, %xmm3
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm4
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm4[0],xmm2[0],xmm4[1],xmm2[1],xmm4[2],xmm2[2],xmm4[3],xmm2[3],xmm4[4],xmm2[4],xmm4[5],xmm2[5],xmm4[6],xmm2[6],xmm4[7],xmm2[7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_32_00_32_00_32_00_32_00_32_00_32_00_32_00_32_16_48_16_48_16_48_16_48_16_48_16_48_16_48_16_48:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpshufb %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,16,17,16,17,16,17,16,17,16,17,16,17,16,17,16,17]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 0, i32 32, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48, i32 16, i32 48>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_32_32_32_32_32_32_32_32_08_09_10_11_12_13_14_15_48_48_48_48_48_48_48_48_24_25_26_27_28_29_30_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_32_32_32_32_32_32_32_32_08_09_10_11_12_13_14_15_48_48_48_48_48_48_48_48_24_25_26_27_28_29_30_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm3
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm3 = xmm3[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm3 = xmm3[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm3[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_32_32_32_32_32_32_32_32_08_09_10_11_12_13_14_15_48_48_48_48_48_48_48_48_24_25_26_27_28_29_30_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %ymm2, %ymm2, %ymm2
; AVX2-NEXT:    vpshufb %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 48, i32 48, i32 48, i32 48, i32 48, i32 48, i32 48, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_39_38_37_36_35_34_33_32_15_14_13_12_11_10_09_08_55_54_53_52_51_50_49_48_31_30_29_28_27_26_25_24(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_39_38_37_36_35_34_33_32_15_14_13_12_11_10_09_08_55_54_53_52_51_50_49_48_31_30_29_28_27_26_25_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <u,u,u,u,u,u,u,u,15,14,13,12,11,10,9,8>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = <7,6,5,4,3,2,1,0,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm4
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm4[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_39_38_37_36_35_34_33_32_15_14_13_12_11_10_09_08_55_54_53_52_51_50_49_48_31_30_29_28_27_26_25_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[u,u,u,u,u,u,u,u,15,14,13,12,11,10,9,8,u,u,u,u,u,u,u,u,31,30,29,28,27,26,25,24]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[7,6,5,4,3,2,1,0,u,u,u,u,u,u,u,u,23,22,21,20,19,18,17,16,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_39_38_37_36_35_34_33_32_07_06_05_04_03_02_01_00_55_54_53_52_51_50_49_48_23_22_21_20_19_18_17_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_39_38_37_36_35_34_33_32_07_06_05_04_03_02_01_00_55_54_53_52_51_50_49_48_23_22_21_20_19_18_17_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <u,u,u,u,u,u,u,u,7,6,5,4,3,2,1,0>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm5 = <7,6,5,4,3,2,1,0,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm5, %xmm4, %xmm4
; AVX1-NEXT:    vpblendw {{.*#+}} xmm2 = xmm4[0,1,2,3],xmm2[4,5,6,7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm5, %xmm1, %xmm1
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_39_38_37_36_35_34_33_32_07_06_05_04_03_02_01_00_55_54_53_52_51_50_49_48_23_22_21_20_19_18_17_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[u,u,u,u,u,u,u,u,7,6,5,4,3,2,1,0,u,u,u,u,u,u,u,u,23,22,21,20,19,18,17,16]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[7,6,5,4,3,2,1,0,u,u,u,u,u,u,u,u,23,22,21,20,19,18,17,16,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5],ymm0[6,7]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 39, i32 38, i32 37, i32 36, i32 35, i32 34, i32 33, i32 32, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 55, i32 54, i32 53, i32 52, i32 51, i32 50, i32 49, i32 48, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_17_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_17_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_17_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,16,16,16,16,16,16,16,16,16,16,16,16,16,16,17,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 17, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_18_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_18_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_18_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,16,16,16,16,16,16,16,16,16,16,16,16,16,18,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 18, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,23,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 23, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,24,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 24, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_30_16_16_16_16_16_16_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_30_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = [0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_30_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,30,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 30, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_31_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_31_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    movl $15, %eax
; AVX1-NEXT:    vmovd %eax, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpshufb %xmm1, %xmm2, %xmm2
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_31_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 31, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpmovzxbw %xmm2, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm4
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3],xmm2[4],xmm4[4],xmm2[5],xmm4[5],xmm2[6],xmm4[6],xmm2[7],xmm4[7]
; AVX1-NEXT:    vpmovzxbw %xmm0, %xmm0
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpunpcklbw {{.*#+}} ymm0 = ymm0[0],ymm1[0],ymm0[1],ymm1[1],ymm0[2],ymm1[2],ymm0[3],ymm1[3],ymm0[4],ymm1[4],ymm0[5],ymm1[5],ymm0[6],ymm1[6],ymm0[7],ymm1[7],ymm0[16],ymm1[16],ymm0[17],ymm1[17],ymm0[18],ymm1[18],ymm0[19],ymm1[19],ymm0[20],ymm1[20],ymm0[21],ymm1[21],ymm0[22],ymm1[22],ymm0[23],ymm1[23]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <8,9,10,11,12,13,14,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm4
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm4[0],xmm2[0],xmm4[1],xmm2[1],xmm4[2],xmm2[2],xmm4[3],xmm2[3],xmm4[4],xmm2[4],xmm4[5],xmm2[5],xmm4[6],xmm2[6],xmm4[7],xmm2[7]
; AVX1-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpunpckhbw {{.*#+}} ymm0 = ymm0[8],ymm1[8],ymm0[9],ymm1[9],ymm0[10],ymm1[10],ymm0[11],ymm1[11],ymm0[12],ymm1[12],ymm0[13],ymm1[13],ymm0[14],ymm1[14],ymm0[15],ymm1[15],ymm0[24],ymm1[24],ymm0[25],ymm1[25],ymm0[26],ymm1[26],ymm0[27],ymm1[27],ymm0[28],ymm1[28],ymm0[29],ymm1[29],ymm0[30],ymm1[30],ymm0[31],ymm1[31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <8,9,10,11,12,13,14,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm4
; AVX1-NEXT:    vpshufb %xmm3, %xmm4, %xmm3
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; AVX1-NEXT:    vpmovzxbw %xmm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_32_01_33_02_34_03_35_04_36_05_37_06_38_07_39_24_56_25_57_26_58_27_59_28_60_29_61_30_62_31_63:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,u,1,u,2,u,3,u,4,u,5,u,6,u,7,u,24,u,25,u,26,u,27,u,28,u,29,u,30,u,31,u]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,0,u,1,u,2,u,3,u,4,u,5,u,6,u,7,u,24,u,25,u,26,u,27,u,28,u,29,u,30,u,31]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55:
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm2 = <8,9,10,11,12,13,14,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm2, %xmm1, %xmm3
; AVX1-NEXT:    vpshufb %xmm2, %xmm0, %xmm2
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpmovzxbw %xmm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm2, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_08_40_09_41_10_42_11_43_12_44_13_45_14_46_15_47_16_48_17_49_18_50_19_51_20_52_21_53_22_54_23_55:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,u,9,u,10,u,11,u,12,u,13,u,14,u,15,u,16,u,17,u,18,u,19,u,20,u,21,u,22,u,23,u]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[u,8,u,9,u,10,u,11,u,12,u,13,u,14,u,15,u,16,u,17,u,18,u,19,u,20,u,21,u,22,u,23]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_17_16_16_16_16_16_16_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_17_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_01_00_16_17_16_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,16,17,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 16, i32 17, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_18_16_16_16_16_16_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_18_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_00_00_00_00_00_02_00_00_16_16_18_16_16_16_16_16_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,16,16,18,16,16,16,16,16,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0, i32 16, i32 16, i32 18, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_00_07_00_00_00_00_00_00_00_16_16_16_16_16_16_16_23_16_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,16,16,16,16,16,16,16,23,16,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 23, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_00_00_00_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_16_16_16_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,24,16,16,16,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 24, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_30_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_30_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_14_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_30_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,16,16,16,30,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 30, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_31(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_31:
; AVX1:       # BB#0:
; AVX1-NEXT:    movl $15, %eax
; AVX1-NEXT:    vmovd %eax, %xmm1
; AVX1-NEXT:    vpshufb %xmm1, %xmm0, %xmm1
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_15_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_16_16_16_16_16_16_16_31:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,31]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 31>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_28_28_28_28_24_24_24_24_20_20_20_20_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_28_28_28_28_24_24_24_24_20_20_20_20_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12_28_28_28_28_24_24_24_24_20_20_20_20_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,28,28,28,28,24,24,24,24,20,20,20,20,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12, i32 28, i32 28, i32 28, i32 28, i32 24, i32 24, i32 24, i32 24, i32 20, i32 20, i32 20, i32 20, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_08_08_08_08_08_08_08_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_08_08_08_08_08_08_08_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_08_08_08_08_08_08_08_08_00_00_00_00_00_00_00_00_16_16_16_16_16_16_16_16_24_24_24_24_24_24_24_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,24,24,24,24,24,24,24,24]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_16_uu_uu_uu_uu_uu_16_16_16_16_16_30_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_16_uu_uu_uu_uu_uu_16_16_16_16_16_30_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,0,0,0,u,u,u,u,u,0,0,0,0,0,14,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_16_uu_uu_uu_uu_uu_16_16_16_16_16_30_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,16,16,16,16,u,u,u,u,u,16,16,16,16,16,30,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 16, i32 16, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 16, i32 16, i32 16, i32 16, i32 30, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_uu_14_uu_uu_00_00_00_00_00_00_00_00_00_00_00_00_16_16_uu_16_uu_uu_uu_uu_16_16_16_16_16_16_30_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_uu_14_uu_uu_00_00_00_00_00_00_00_00_00_00_00_00_16_16_uu_16_uu_uu_uu_uu_16_16_16_16_16_16_30_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[14,14,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,u,0,u,u,u,u,0,0,0,0,0,0,14,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_uu_14_uu_uu_00_00_00_00_00_00_00_00_00_00_00_00_16_16_uu_16_uu_uu_uu_uu_16_16_16_16_16_16_30_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[u,14,u,u,0,0,0,0,0,0,0,0,0,0,0,0,16,16,u,16,u,u,u,u,16,16,16,16,16,16,30,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 undef, i32 14, i32 undef, i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 16, i32 16, i32 undef, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 30, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_00_00_00_uu_uu_uu_04_uu_08_08_08_08_uu_uu_12_uu_28_28_28_28_uu_uu_uu_24_20_20_20_20_16_16_16_16(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_00_00_00_uu_uu_uu_04_uu_08_08_08_08_uu_uu_12_uu_28_28_28_28_uu_uu_uu_24_20_20_20_20_16_16_16_16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm0[0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_00_00_00_uu_uu_uu_04_uu_08_08_08_08_uu_uu_12_uu_28_28_28_28_uu_uu_uu_24_20_20_20_20_16_16_16_16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[0,0,0,u,u,u,4,u,8,8,8,8,u,u,12,u,28,28,28,28,u,u,u,24,20,20,20,20,16,16,16,16]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 0, i32 0, i32 0, i32 undef, i32 undef, i32 undef, i32 4, i32 undef, i32 8, i32 8, i32 8, i32 8, i32 undef, i32 undef, i32 12, i32 undef, i32 28, i32 28, i32 28, i32 28, i32 undef, i32 undef, i32 undef, i32 24, i32 20, i32 20, i32 20, i32 20, i32 16, i32 16, i32 16, i32 16>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_08_08_08_08_08_08_08_08_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_uu_uu_uu_uu_uu_uu_uu_24_24_24_24_24_24(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_08_08_08_08_08_08_08_08_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_uu_uu_uu_uu_uu_uu_uu_24_24_24_24_24_24:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpunpckhbw {{.*#+}} xmm1 = xmm0[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; AVX1-NEXT:    vpshuflw {{.*#+}} xmm1 = xmm1[0,0,0,0,4,5,6,7]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,0,0,0,8,8,9,9,8,8,8,8,8,8,8,8]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_08_08_08_08_08_08_08_08_uu_uu_uu_uu_uu_uu_uu_uu_16_16_16_uu_uu_uu_uu_uu_uu_uu_24_24_24_24_24_24:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[8,8,8,8,8,8,8,8,u,u,u,u,u,u,u,u,16,16,16,u,u,u,u,u,u,u,24,24,24,24,24,24]
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 16, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  ret <32 x i8> %shuffle
}

define <32 x i8> @shuffle_v32i8_42_45_12_13_35_35_60_40_17_22_29_44_33_12_48_51_20_19_52_19_49_54_37_32_48_42_59_07_36_34_36_39(<32 x i8> %a, <32 x i8> %b) {
; AVX1-LABEL: shuffle_v32i8_42_45_12_13_35_35_60_40_17_22_29_44_33_12_48_51_20_19_52_19_49_54_37_32_48_42_59_07_36_34_36_39:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = zero,zero,xmm0[u],zero,xmm0[u,u,u,u,u,u,u,7,u,u,u,u]
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm3
; AVX1-NEXT:    vpshufb {{.*#+}} xmm4 = xmm3[4,3,u,3,u,u,u,u,u,u,u],zero,xmm3[u,u,u,u]
; AVX1-NEXT:    vpor %xmm2, %xmm4, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm2 = xmm2[0,1],zero,xmm2[3],zero,zero,zero,zero,zero,zero,zero,xmm2[11],zero,zero,zero,zero
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm4
; AVX1-NEXT:    vpshufb {{.*#+}} xmm5 = xmm4[u,u,4,u,1,6],zero,zero,xmm4[0],zero,xmm4[11,u],zero,zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm6 = xmm1[u,u],zero,xmm1[u],zero,zero,xmm1[5,0],zero,xmm1[10],zero,xmm1[u,4,2,4,7]
; AVX1-NEXT:    vpor %xmm5, %xmm6, %xmm5
; AVX1-NEXT:    vpshufb {{.*#+}} xmm5 = zero,zero,xmm5[2],zero,xmm5[4,5,6,7,8,9,10],zero,xmm5[12,13,14,15]
; AVX1-NEXT:    vpor %xmm2, %xmm5, %xmm2
; AVX1-NEXT:    vpshufb {{.*#+}} xmm3 = xmm3[u,u],zero,zero,xmm3[u,u,u,u,1,6,13,u,u],zero,xmm3[u,u]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[u,u,12,13,u,u,u,u],zero,zero,zero,xmm0[u,u,12,u,u]
; AVX1-NEXT:    vpor %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = zero,zero,xmm0[2,3],zero,zero,zero,zero,xmm0[8,9,10],zero,zero,xmm0[13],zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm3 = zero,zero,xmm4[u,u],zero,zero,xmm4[12],zero,xmm4[u,u,u],zero,zero,xmm4[u,0,3]
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[10,13,u,u,3,3],zero,xmm1[8,u,u,u,12,1,u],zero,zero
; AVX1-NEXT:    vpor %xmm3, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb {{.*#+}} xmm1 = xmm1[0,1],zero,zero,xmm1[4,5,6,7],zero,zero,zero,xmm1[11,12],zero,xmm1[14,15]
; AVX1-NEXT:    vpor %xmm0, %xmm1, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuffle_v32i8_42_45_12_13_35_35_60_40_17_22_29_44_33_12_48_51_20_19_52_19_49_54_37_32_48_42_59_07_36_34_36_39:
; AVX2:       # BB#0:
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm2 = ymm1[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm2 = ymm2[u,u,u,u,u,u,12,u,u,u,u,u,u,u,0,3,u,u,u,u,u,u,21,16,u,26,u,u,20,18,20,23]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm1 = ymm1[10,13,u,u,3,3,u,8,u,u,u,12,1,u,u,u,u,u,20,u,17,22,u,u,16,u,27,u,u,u,u,u]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm3 = <255,255,u,u,255,255,0,255,u,u,u,255,255,u,0,0,u,u,255,u,255,255,0,0,255,0,255,u,0,0,0,0>
; AVX2-NEXT:    vpblendvb %ymm3, %ymm1, %ymm2, %ymm1
; AVX2-NEXT:    vperm2i128 {{.*#+}} ymm2 = ymm0[2,3,0,1]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm2 = ymm2[u,u,u,u,u,u,u,u,1,6,13,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,23,u,u,u,u]
; AVX2-NEXT:    vpshufb {{.*#+}} ymm0 = ymm0[u,u,12,13,u,u,u,u,u,u,u,u,u,12,u,u,20,19,u,19,u,u,u,u,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm2[2],ymm0[3,4,5],ymm2[6],ymm0[7]
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [255,255,0,0,255,255,255,255,0,0,0,255,255,0,255,255,0,0,255,0,255,255,255,255,255,255,255,0,255,255,255,255]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %shuffle = shufflevector <32 x i8> %a, <32 x i8> %b, <32 x i32> <i32 42, i32 45, i32 12, i32 13, i32 35, i32 35, i32 60, i32 40, i32 17, i32 22, i32 29, i32 44, i32 33, i32 12, i32 48, i32 51, i32 20, i32 19, i32 52, i32 19, i32 49, i32 54, i32 37, i32 32, i32 48, i32 42, i32 59, i32 7, i32 36, i32 34, i32 36, i32 39>
  ret <32 x i8> %shuffle
}
