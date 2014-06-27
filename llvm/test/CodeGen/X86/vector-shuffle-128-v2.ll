; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=CHECK-SSE2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <2 x i64> @shuffle_v2i64_00(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_00
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_10(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_10
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[2,3,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_11(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_11
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_22(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_22
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[0,1,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_32(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_32
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[2,3,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_33(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_33
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 3>
  ret <2 x i64> %shuffle
}

define <2 x double> @shuffle_v2f64_00(<2 x double> %a, <2 x double> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2f64_00
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[0,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_10(<2 x double> %a, <2 x double> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2f64_10
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[1,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_11(<2 x double> %a, <2 x double> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2f64_11
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[1,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_22(<2 x double> %a, <2 x double> %b) {
; FIXME: Should these use movapd + shufpd to remove a domain change at the cost
;        of a mov?
;
; CHECK-SSE2-LABEL: @shuffle_v2f64_22
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[0,1,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 2>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_32(<2 x double> %a, <2 x double> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2f64_32
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[2,3,0,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 3, i32 2>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_33(<2 x double> %a, <2 x double> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2f64_33
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 3, i32 3>
  ret <2 x double> %shuffle
}


define <2 x i64> @shuffle_v2i64_02(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_02
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_02_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_02_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm2[0]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_03(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_03
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_03_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_03_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_12(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_12
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[1],xmm1[0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_12_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_12_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[1],xmm2[0]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_13(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_13
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[1],xmm1[1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_13_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_13_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[1],xmm2[1]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_20(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_20
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[0]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_20_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_20_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm2 = xmm2[0],xmm1[0]
; CHECK-SSE2-NEXT:    movapd %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_21(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_21
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_21_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_21_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; CHECK-SSE2-NEXT:    movapd %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_30(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_30
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[1],xmm0[0]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_30_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_30_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm2 = xmm2[1],xmm1[0]
; CHECK-SSE2-NEXT:    movapd %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_31(<2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_31
; CHECK-SSE2:         shufpd {{.*}} # xmm1 = xmm1[1],xmm0[1]
; CHECK-SSE2-NEXT:    movapd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_31_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; CHECK-SSE2-LABEL: @shuffle_v2i64_31_copy
; CHECK-SSE2:         shufpd {{.*}} # xmm2 = xmm2[1],xmm1[1]
; CHECK-SSE2-NEXT:    movapd %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 1>
  ret <2 x i64> %shuffle
}
