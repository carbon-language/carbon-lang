; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=AVX1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <4 x i64> @shuffle_v4i64_0001(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0001
; AVX1:       # BB#0:
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0020(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0020
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0112(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0112
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm0[1],xmm1[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 1, i32 2>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_0300(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0300
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm0[0],xmm1[1]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_1000(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_1000
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_2200(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_2200
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0,0]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_3330(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_3330
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[1],xmm0[0]
; AVX1-NEXT:    vpunpckhqdq {{.*}} # xmm1 = xmm1[1,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x i64> @shuffle_v4i64_3210(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_3210
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*}} # xmm1 = xmm1[2,3,0,1]
; AVX1-NEXT:    vpshufd {{.*}} # xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i64> %shuffle
}

define <4 x double> @shuffle_v4f64_0001(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_0001
; AVX1:       # BB#0:
; AVX1-NEXT:    vmovlhps {{.*}} # xmm1 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_0020(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_0020
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; AVX1-NEXT:    vmovlhps {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_0300(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_0300
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm0[0],xmm1[1]
; AVX1-NEXT:    vmovlhps {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_1000(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_1000
; AVX1:       # BB#0:
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm0[1,0]
; AVX1-NEXT:    vmovlhps {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_2200(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_2200
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vmovlhps {{.*}} # xmm1 = xmm1[0,0]
; AVX1-NEXT:    vmovlhps {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_3330(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_3330
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[1],xmm0[0]
; AVX1-NEXT:    vmovhlps {{.*}} # xmm1 = xmm1[1,1]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x double> %shuffle
}
define <4 x double> @shuffle_v4f64_3210(<4 x double> %a, <4 x double> %b) {
; AVX1-LABEL: @shuffle_v4f64_3210
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm1[1,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm0[1,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x double> %shuffle
}

define <4 x i64> @shuffle_v4i64_0124(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0124
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm2[0],xmm1[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_0142(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0142
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm2 = xmm2[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_0412(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0412
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vshufpd {{.*}} # xmm2 = xmm0[1],xmm2[0]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 2>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_4012(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_4012
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vshufpd {{.*}} # xmm2 = xmm0[1],xmm2[0]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[0],xmm0[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 2>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_0145(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0145
; AVX1:       # BB#0:
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_0451(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_0451
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*}} # xmm2 = xmm1[2,3,0,1]
; AVX1-NEXT:    vshufpd {{.*}} # xmm2 = xmm2[0],xmm0[1]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm1 = xmm1[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 4, i32 5, i32 1>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_4501(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_4501
; AVX1:       # BB#0:
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 5, i32 0, i32 1>
  ret <4 x i64> %shuffle
}
define <4 x i64> @shuffle_v4i64_4015(<4 x i64> %a, <4 x i64> %b) {
; AVX1-LABEL: @shuffle_v4i64_4015
; AVX1:       # BB#0:
; AVX1-NEXT:    vpshufd {{.*}} # xmm2 = xmm0[2,3,0,1]
; AVX1-NEXT:    vshufpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; AVX1-NEXT:    vpunpcklqdq {{.*}} # xmm0 = xmm0[0,0]
; AVX1-NEXT:    vshufpd {{.*}} # xmm0 = xmm1[0],xmm0[1]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 5>
  ret <4 x i64> %shuffle
}
