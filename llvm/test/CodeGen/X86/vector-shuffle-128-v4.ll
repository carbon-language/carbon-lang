; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE41
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
define <4 x float> @shuffle_v4f32_0022(<4 x float> %a, <4 x float> %b) {
; SSE2-LABEL: @shuffle_v4f32_0022
; SSE2:         shufps {{.*}} # xmm0 = xmm0[0,0,2,2]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_0022
; SSE41:         movsldup {{.*}} # xmm0 = xmm0[0,0,2,2]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_0022
; AVX1:         vmovsldup {{.*}} # xmm0 = xmm0[0,0,2,2]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x float> %shuffle
}
define <4 x float> @shuffle_v4f32_1133(<4 x float> %a, <4 x float> %b) {
; SSE2-LABEL: @shuffle_v4f32_1133
; SSE2:         shufps {{.*}} # xmm0 = xmm0[1,1,3,3]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_1133
; SSE41:         movshdup {{.*}} # xmm0 = xmm0[1,1,3,3]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_1133
; AVX1:         vmovshdup {{.*}} # xmm0 = xmm0[1,1,3,3]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  ret <4 x float> %shuffle
}

define <4 x i32> @shuffle_v4i32_0124(<4 x i32> %a, <4 x i32> %b) {
; SSE2-LABEL: @shuffle_v4i32_0124
; SSE2:         shufps {{.*}} # xmm1 = xmm1[0,0],xmm0[2,0]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,1],xmm1[2,0]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4i32_0124
; SSE41:         insertps {{.*}} # xmm0 = xmm0[0,1,2],xmm1[0]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4i32_0124
; AVX1:         vinsertps {{.*}} # xmm0 = xmm0[0,1,2],xmm1[0]
; AVX1-NEXT:    retq
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

define <4 x float> @shuffle_v4f32_4zzz(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_4zzz
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,0],[[X]][1,0]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,2],[[X]][2,3]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_4zzz
; SSE41:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE41-NEXT:    blendps {{.*}} # [[X]] = xmm0[0],[[X]][1,2,3]
; SSE41-NEXT:    movaps %[[X]], %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_4zzz
; AVX1:         vxorps %[[X:xmm[0-9]+]], %[[X]]
; AVX1-NEXT:    vblendps {{.*}} # xmm0 = xmm0[0],[[X]][1,2,3]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_z4zz(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_z4zz
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,0],[[X]][2,0]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[2,0],[[X]][3,0]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_z4zz
; SSE41:         insertps {{.*}} # xmm0 = zero,xmm0[0],zero,zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_z4zz
; AVX1:         vinsertps {{.*}} # xmm0 = zero,xmm0[0],zero,zero
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 2, i32 4, i32 3, i32 0>
  ret <4 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_zz4z(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_zz4z
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[0,0],[[X]][0,0]
; SSE2-NEXT:    shufps {{.*}} # [[X]] = [[X]][0,0],xmm0[0,2]
; SSE2-NEXT:    movaps %[[X]], %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_zz4z
; SSE41:         insertps {{.*}} # xmm0 = zero,zero,xmm0[0],zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_zz4z
; AVX1:         vinsertps {{.*}} # xmm0 = zero,zero,xmm0[0],zero
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 0, i32 0, i32 4, i32 0>
  ret <4 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_zuu4(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_zuu4
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # [[X]] = [[X]][0,1],xmm0[2,0]
; SSE2-NEXT:    movaps %[[X]], %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_zuu4
; SSE41:         insertps {{.*}} # xmm0 = zero,zero,zero,xmm0[0]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_zuu4
; AVX1:         vinsertps {{.*}} # xmm0 = zero,zero,zero,xmm0[0]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 0, i32 undef, i32 undef, i32 4>
  ret <4 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_zzz7(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_zzz7
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[3,0],[[X]][2,0]
; SSE2-NEXT:    shufps {{.*}} # [[X]] = [[X]][0,1],xmm0[2,0]
; SSE2-NEXT:    movaps %[[X]], %xmm0
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_zzz7
; SSE41:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE41-NEXT:    blendps {{.*}} # [[X]] = [[X]][0,1,2],xmm0[3]
; SSE41-NEXT:    movaps %[[X]], %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_zzz7
; AVX1:         vxorps %[[X:xmm[0-9]+]], %[[X]]
; AVX1-NEXT:    vblendps {{.*}} # xmm0 = [[X]][0,1,2],xmm0[3]
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %shuffle
}

define <4 x float> @shuffle_v4f32_z6zz(<4 x float> %a) {
; SSE2-LABEL: @shuffle_v4f32_z6zz
; SSE2:         xorps %[[X:xmm[0-9]+]], %[[X]]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[2,0],[[X]][0,0]
; SSE2-NEXT:    shufps {{.*}} # xmm0 = xmm0[2,0],[[X]][2,3]
; SSE2-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v4f32_z6zz
; SSE41:         insertps {{.*}} # xmm0 = zero,xmm0[2],zero,zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: @shuffle_v4f32_z6zz
; AVX1:         vinsertps {{.*}} # xmm0 = zero,xmm0[2],zero,zero
; AVX1-NEXT:    retq
  %shuffle = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x float> %shuffle
}

define <4 x i32> @shuffle_v4i32_4zzz(i32 %i) {
; ALL-LABEL: @shuffle_v4i32_4zzz
; ALL:         movd {{.*}}, %xmm0
; ALL-NEXT:    retq
  %a = insertelement <4 x i32> undef, i32 %i, i32 0
  %shuffle = shufflevector <4 x i32> zeroinitializer, <4 x i32> %a, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_z4zz(i32 %i) {
; ALL-LABEL: @shuffle_v4i32_z4zz
; ALL:         movd {{.*}}, %xmm0
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[1,0,1,1]
; ALL-NEXT:    retq
  %a = insertelement <4 x i32> undef, i32 %i, i32 0
  %shuffle = shufflevector <4 x i32> zeroinitializer, <4 x i32> %a, <4 x i32> <i32 2, i32 4, i32 3, i32 0>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_zz4z(i32 %i) {
; ALL-LABEL: @shuffle_v4i32_zz4z
; ALL:         movd {{.*}}, %xmm0
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[1,1,0,1]
; ALL-NEXT:    retq
  %a = insertelement <4 x i32> undef, i32 %i, i32 0
  %shuffle = shufflevector <4 x i32> zeroinitializer, <4 x i32> %a, <4 x i32> <i32 0, i32 0, i32 4, i32 0>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_zuu4(i32 %i) {
; ALL-LABEL: @shuffle_v4i32_zuu4
; ALL:         movd {{.*}}, %xmm0
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[1,1,1,0]
; ALL-NEXT:    retq
  %a = insertelement <4 x i32> undef, i32 %i, i32 0
  %shuffle = shufflevector <4 x i32> zeroinitializer, <4 x i32> %a, <4 x i32> <i32 0, i32 undef, i32 undef, i32 4>
  ret <4 x i32> %shuffle
}

define <4 x i32> @shuffle_v4i32_z6zz(i32 %i) {
; ALL-LABEL: @shuffle_v4i32_z6zz
; ALL:         movd {{.*}}, %xmm0
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[1,0,1,1]
; ALL-NEXT:    retq
  %a = insertelement <4 x i32> undef, i32 %i, i32 2
  %shuffle = shufflevector <4 x i32> zeroinitializer, <4 x i32> %a, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x i32> %shuffle
}
