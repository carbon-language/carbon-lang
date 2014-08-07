; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=CHECK-SSE2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <4 x i32> @shuffle_v4i32_0001(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0001
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,0,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0020(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0020
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,0,2,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0112(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0112
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,1,1,2]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0300(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0300
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,3,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_1000(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_1000
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[1,0,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_2200(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_2200
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[2,2,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_3330(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_3330
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[3,3,3,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_3210(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_3210
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[3,2,1,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %shuffle
}

define <4 x float> @shuffle_v4f32_0001(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_0001
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[0,0,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_0020(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_0020
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[0,0,2,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_0300(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_0300
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[0,3,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 3, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_1000(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_1000
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[1,0,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_2200(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_2200
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[2,2,0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 2, i32 2, i32 0, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_3330(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_3330
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[3,3,3,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 0>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_3210(<4 x float> %a, <4 x float> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4f32_3210
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[3,2,1,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x float> %shuffle
}

define <4 x i32> @shuffle_v4i32_0124(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0124
; CHECK-SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[2,0]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[2,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0142(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0142
; CHECK-SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[2,0]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,2]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0412(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0412
; CHECK-SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm1 = xmm1[2,0],xmm0[1,2]
; CHECK-SSE2-NEXT:    movaps %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4012(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_4012
; CHECK-SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[0,0]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm1 = xmm1[0,2],xmm0[1,2]
; CHECK-SSE2-NEXT:    movaps %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 2>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0145(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0145
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_0451(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_0451
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,1]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,2,3,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 5, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4501(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_4501
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 5, i32 0, i32 1>
  ret <4 x i32> %shuffle
}
define <4 x i32> @shuffle_v4i32_4015(<4 x i32> %a, <4 x i32> %b) {
; CHECK-SSE2-LABEL: @shuffle_v4i32_4015
; CHECK-SSE2:         shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[0,1]
; CHECK-SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[2,0,1,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 0, i32 1, i32 5>
  ret <4 x i32> %shuffle
}
