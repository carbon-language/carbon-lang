; RUN: llc < %s -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <4 x i64> @shuffle_v4i64_0001(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0001
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0020(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0020
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0112(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0112
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufpd {{.*}} # xmm1 = xmm0[1],xmm1[0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 1, i32 2>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0300(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0300
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vblendpd {{.*}} # xmm1 = xmm0[0],xmm1[1]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_1000(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_1000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_2200(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_2200
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_3330(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_3330
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[1],xmm0[0]
; ALL-NEXT:    vmovhlps {{.*}} # xmm1 = xmm1[1,1]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_3210(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_3210
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x double> @shuffle_v4f64_0001(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0001
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0020(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0020
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0300(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0300
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vblendpd {{.*}} # xmm1 = xmm0[0],xmm1[1]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1000(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1000
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm0[1,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_2200(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_2200
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_3330(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_3330
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[1],xmm0[0]
; ALL-NEXT:    vmovhlps {{.*}} # xmm1 = xmm1[1,1]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_3210(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_3210
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm1
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0023(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0023
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[0,0,2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0022(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0022
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[0,0,2,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1032(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1032
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[1,0,3,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1133(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1133
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[1,1,3,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1023(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1023
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[1,0,2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 0, i32 2, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1022(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1022
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[1,0,2,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 0, i32 2, i32 2>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0423(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0423
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm1 = ymm1[0,0,2,2]
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1],ymm0[2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0462(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0462
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # ymm1 = ymm1[0,0,2,2]
; ALL-NEXT:    vpermilpd {{.*}} # ymm0 = ymm0[0,0,2,2]
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1,2],ymm0[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 4, i32 6, i32 2>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0426(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0426
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # ymm0 = ymm0[0],ymm1[0],ymm0[2],ymm1[2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_1537(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_1537
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhpd {{.*}} # ymm0 = ymm0[1],ymm1[1],ymm0[3],ymm1[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_4062(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_4062
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*}} # ymm0 = ymm1[0],ymm0[0],ymm1[2],ymm0[2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 4, i32 0, i32 6, i32 2>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_5173(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_5173
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhpd {{.*}} # ymm0 = ymm1[1],ymm0[1],ymm1[3],ymm0[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 5, i32 1, i32 7, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_5163(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_5163
; ALL:       # BB#0:
; ALL-NEXT:    vshufpd {{.*}} # ymm0 = ymm1[1],ymm0[1],ymm1[2],ymm0[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 5, i32 1, i32 6, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0527(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0527
; ALL:       # BB#0:
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_4163(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_4163
; ALL:       # BB#0:
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm1[0],ymm0[1],ymm1[2],ymm0[3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0145(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0145
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_4501(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_4501
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 4, i32 5, i32 0, i32 1>
  ret <4 x double> %shuffle
}

define <4 x double> @shuffle_v4f64_0167(<4 x double> %a, <4 x double> %b) {
; ALL-LABEL: @shuffle_v4f64_0167
; ALL:       # BB#0:
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0,1],ymm1[2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x double> %shuffle
}

define <4 x i64> @shuffle_v4i64_0124(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0124
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm1 = xmm2[0],xmm1[1]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0142(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0142
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vunpcklpd {{.*}} # xmm2 = xmm2[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0412(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0412
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vshufpd {{.*}} # xmm2 = xmm0[1],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 2>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_4012(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_4012
; ALL:       # BB#0:
; ALL-NEXT:    vextractf128 $1, %ymm0, %xmm2
; ALL-NEXT:    vshufpd {{.*}} # xmm2 = xmm0[1],xmm2[0]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm0 = xmm1[0],xmm0[1]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 2>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0145(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0145
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0451(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_0451
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm1[1,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm2 = xmm2[0],xmm0[1]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm1 = xmm1[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 5, i32 1>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_4501(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_4501
; ALL:       # BB#0:
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 5, i32 0, i32 1>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_4015(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @shuffle_v4i64_4015
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm2 = xmm0[1,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; ALL-NEXT:    vunpcklpd {{.*}} # xmm0 = xmm0[0,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm0 = xmm1[0],xmm0[1]
; ALL-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 5>
  ret <4 x i64> %shuffle
}

define <4 x i64> @stress_test1(<4 x i64> %a, <4 x i64> %b) {
; ALL-LABEL: @stress_test1
; ALL:       # BB#0:
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm1[1,0]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vextractf128 $1, %ymm1, %xmm1
; ALL-NEXT:    vmovhlps {{.*}} # xmm1 = xmm1[1,1]
; ALL-NEXT:    vpermilpd {{.*}} # xmm1 = xmm1[1,0]
; ALL-NEXT:    vblendpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; ALL-NEXT:    vpermilpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; ALL-NEXT:    retq
  %c = shufflevector <4 x i64> %b, <4 x i64> undef, <4 x i32> <i32 3, i32 1, i32 1, i32 0>
  %d = shufflevector <4 x i64> %c, <4 x i64> undef, <4 x i32> <i32 3, i32 undef, i32 2, i32 undef>
  %e = shufflevector <4 x i64> %b, <4 x i64> undef, <4 x i32> <i32 3, i32 3, i32 1, i32 undef>
  %f = shufflevector <4 x i64> %d, <4 x i64> %e, <4 x i32> <i32 5, i32 1, i32 1, i32 0>

  ret <4 x i64> %f
}

define <4 x i64> @insert_reg_and_zero_v4i64(i64 %a) {
; ALL-LABEL: @insert_reg_and_zero_v4i64
; ALL:       # BB#0:
; ALL-NEXT:    vmovq %rdi, %xmm0
; ALL-NEXT:    vxorpd %ymm1, %ymm1, %ymm1
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    retq
  %v = insertelement <4 x i64> undef, i64 %a, i64 0
  %shuffle = shufflevector <4 x i64> %v, <4 x i64> zeroinitializer, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x i64> %shuffle
}

define <4 x i64> @insert_mem_and_zero_v4i64(i64* %ptr) {
; ALL-LABEL: @insert_mem_and_zero_v4i64
; ALL:       # BB#0:
; ALL-NEXT:    vmovq (%rdi), %xmm0
; ALL-NEXT:    vxorpd %ymm1, %ymm1, %ymm1
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    retq
  %a = load i64* %ptr
  %v = insertelement <4 x i64> undef, i64 %a, i64 0
  %shuffle = shufflevector <4 x i64> %v, <4 x i64> zeroinitializer, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x i64> %shuffle
}

define <4 x double> @insert_reg_and_zero_v4f64(double %a) {
; ALL-LABEL: @insert_reg_and_zero_v4f64
; ALL:       # BB#0:
; ALL:         vxorpd %ymm1, %ymm1, %ymm1
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    retq
  %v = insertelement <4 x double> undef, double %a, i32 0
  %shuffle = shufflevector <4 x double> %v, <4 x double> zeroinitializer, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x double> %shuffle
}

define <4 x double> @insert_mem_and_zero_v4f64(double* %ptr) {
; ALL-LABEL: @insert_mem_and_zero_v4f64
; ALL:       # BB#0:
; ALL-NEXT:    vmovsd (%rdi), %xmm0
; ALL-NEXT:    vxorpd %ymm1, %ymm1, %ymm1
; ALL-NEXT:    vblendpd {{.*}} # ymm0 = ymm0[0],ymm1[1,2,3]
; ALL-NEXT:    retq
  %a = load double* %ptr
  %v = insertelement <4 x double> undef, double %a, i32 0
  %shuffle = shufflevector <4 x double> %v, <4 x double> zeroinitializer, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x double> %shuffle
}
