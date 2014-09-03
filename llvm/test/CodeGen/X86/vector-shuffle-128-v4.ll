; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=AVX1

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <4 x i32> @shuffle_v4i32_0001(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0001
; ALL:         pshufd {{.*}} # xmm0 = xmm0[0,0,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0020(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0020
; ALL:         pshufd {{.*}} # xmm0 = xmm0[0,0,2,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0112(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0112
; ALL:         pshufd {{.*}} # xmm0 = xmm0[0,1,1,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0300(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0300
; ALL:         pshufd {{.*}} # xmm0 = xmm0[0,3,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_1000(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_1000
; ALL:         pshufd {{.*}} # xmm0 = xmm0[1,0,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_2200(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_2200
; ALL:         pshufd {{.*}} # xmm0 = xmm0[2,2,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_3330(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_3330
; ALL:         pshufd {{.*}} # xmm0 = xmm0[3,3,3,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_3210(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_3210
; ALL:         pshufd {{.*}} # xmm0 = xmm0[3,2,1,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_2121(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_2121
; ALL:         pshufd {{.*}} # xmm0 = xmm0[2,1,2,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 1, i32 2, i32 1>
  ret <4 x i32> %shuffle
}

define <4 x float> @shuffle_v4f32_0001(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_0001
; ALL:         shufps {{.*}} # xmm0 = xmm0[0,0,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_0020(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_0020
; ALL:         shufps {{.*}} # xmm0 = xmm0[0,0,2,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_0300(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_0300
; ALL:         shufps {{.*}} # xmm0 = xmm0[0,3,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_1000(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_1000
; ALL:         shufps {{.*}} # xmm0 = xmm0[1,0,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_2200(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_2200
; ALL:         shufps {{.*}} # xmm0 = xmm0[2,2,0,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_3330(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_3330
; ALL:         shufps {{.*}} # xmm0 = xmm0[3,3,3,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_3210(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: @shuffle_v4f32_3210
; ALL:         shufps {{.*}} # xmm0 = xmm0[3,2,1,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %shuffle
}

define <4 x i32> @shuffle_v4i32_0124(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0124
; ALL:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[2,0]
; ALL-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[2,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0142(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0142
; ALL:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[2,0]
; ALL-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0412(<4 x i32> %a, <4 x i32> %b) {
; SSE2-LABEL: @shuffle_v4i32_0412
; SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; SSE2-NEXT:    shufps {{.*}} # xmm1 = xmm1[2,0],xmm0[1,2]
; SSE2-NEXT:    movaps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4i32_0412
; AVX1:         vshufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; AVX1-NEXT:    vshufps {{.*}} # xmm0 = xmm1[2,0],xmm0[1,2]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4012(<4 x i32> %a, <4 x i32> %b) {
; SSE2-LABEL: @shuffle_v4i32_4012
; SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; SSE2-NEXT:    shufps {{.*}} # xmm1 = xmm1[0,2],xmm0[1,2]
; SSE2-NEXT:    movaps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4i32_4012
; AVX1:         vshufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; AVX1-NEXT:    vshufps {{.*}} # xmm0 = xmm1[0,2],xmm0[1,2]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0145(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0145
; ALL:         punpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0451(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_0451
; ALL:         shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,1]
; ALL-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,2,3,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 5, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4501(<4 x i32> %a, <4 x i32> %b) {
; SSE2-LABEL: @shuffle_v4i32_4501
; SSE2:         punpcklqdq {{.*}} # xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4i32_4501
; AVX1:         punpcklqdq {{.*}} # xmm0 = xmm1[0],xmm0[0]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 5, i32 0, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4015(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: @shuffle_v4i32_4015
; ALL:         shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,1]
; ALL-NEXT:    shufps {{.*}} # xmm0 = xmm0[2,0,1,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 5>
  ret <4 x i32> %shuffle
}
