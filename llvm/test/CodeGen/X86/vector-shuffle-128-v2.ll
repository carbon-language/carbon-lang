; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse3 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE41

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <2 x i64> @shuffle_v2i64_00(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_00
; ALL:         pshufd {{.*}} # xmm0 = xmm0[0,1,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_10(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_10
; ALL:         pshufd {{.*}} # xmm0 = xmm0[2,3,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_11(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_11
; ALL:         pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_22(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_22
; ALL:         pshufd {{.*}} # xmm0 = xmm1[0,1,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_32(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_32
; ALL:         pshufd {{.*}} # xmm0 = xmm1[2,3,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_33(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_33
; ALL:         pshufd {{.*}} # xmm0 = xmm1[2,3,2,3]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 3>
  ret <2 x i64> %shuffle
}

define <2 x double> @shuffle_v2f64_00(<2 x double> %a, <2 x double> %b) {
; SSE2-LABEL: @shuffle_v2f64_00
; SSE2:         movlhps {{.*}} # xmm0 = xmm0[0,0]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2f64_00
; SSE3:         unpcklpd {{.*}} # xmm0 = xmm0[0,0]
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2f64_00
; SSE41:         unpcklpd {{.*}} # xmm0 = xmm0[0,0]
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_10(<2 x double> %a, <2 x double> %b) {
; ALL-LABEL: @shuffle_v2f64_10
; ALL:         shufpd {{.*}} # xmm0 = xmm0[1,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_11(<2 x double> %a, <2 x double> %b) {
; ALL-LABEL: @shuffle_v2f64_11
; ALL:         movhlps {{.*}} # xmm0 = xmm0[1,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_22(<2 x double> %a, <2 x double> %b) {
; SSE2-LABEL: @shuffle_v2f64_22
; SSE2:         movlhps {{.*}} # xmm1 = xmm1[0,0]
; SSE2-NEXT:    movaps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2f64_22
; SSE3:         unpcklpd {{.*}} # xmm1 = xmm1[0,0]
; SSE3-NEXT:    movapd %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2f64_22
; SSE41:         unpcklpd {{.*}} # xmm1 = xmm1[0,0]
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 2>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_32(<2 x double> %a, <2 x double> %b) {
; ALL-LABEL: @shuffle_v2f64_32
; ALL:         pshufd {{.*}} # xmm0 = xmm1[2,3,0,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 3, i32 2>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_33(<2 x double> %a, <2 x double> %b) {
; ALL-LABEL: @shuffle_v2f64_33
; ALL:         movhlps {{.*}} # xmm1 = xmm1[1,1]
; ALL-NEXT:    movaps %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 3, i32 3>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_03(<2 x double> %a, <2 x double> %b) {
; SSE2-LABEL: @shuffle_v2f64_03
; SSE2:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2f64_03
; SSE3:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2f64_03
; SSE41:         blendpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x double> %shuffle
}
define <2 x double> @shuffle_v2f64_21(<2 x double> %a, <2 x double> %b) {
; SSE2-LABEL: @shuffle_v2f64_21
; SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2f64_21
; SSE3:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE3-NEXT:    movapd %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2f64_21
; SSE41:         blendpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x double> %a, <2 x double> %b, <2 x i32> <i32 2, i32 1>
  ret <2 x double> %shuffle
}


define <2 x i64> @shuffle_v2i64_02(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_02
; ALL:         punpcklqdq {{.*}} # xmm0 = xmm0[0],xmm1[0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_02_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_02_copy
; ALL:         punpcklqdq {{.*}} # xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_03(<2 x i64> %a, <2 x i64> %b) {
; SSE2-LABEL: @shuffle_v2i64_03
; SSE2:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2i64_03
; SSE3:         shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2i64_03
; SSE41:         blendpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_03_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; SSE2-LABEL: @shuffle_v2i64_03_copy
; SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2i64_03_copy
; SSE3:         shufpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; SSE3-NEXT:    movapd %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2i64_03_copy
; SSE41:         blendpd {{.*}} # xmm1 = xmm1[0],xmm2[1]
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_12(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_12
; ALL:         shufpd {{.*}} # xmm0 = xmm0[1],xmm1[0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_12_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_12_copy
; ALL:         shufpd {{.*}} # xmm1 = xmm1[1],xmm2[0]
; ALL-NEXT:    movapd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 2>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_13(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_13
; ALL:         punpckhqdq {{.*}} # xmm0 = xmm0[1],xmm1[1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_13_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_13_copy
; ALL:         punpckhqdq {{.*}} # xmm1 = xmm1[1],xmm2[1]
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_20(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_20
; ALL:         punpcklqdq {{.*}} # xmm1 = xmm1[0],xmm0[0]
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_20_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_20_copy
; ALL:         punpcklqdq {{.*}} # xmm2 = xmm2[0],xmm1[0]
; ALL-NEXT:    movdqa %xmm2, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_21(<2 x i64> %a, <2 x i64> %b) {
; SSE2-LABEL: @shuffle_v2i64_21
; SSE2:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2i64_21
; SSE3:         shufpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE3-NEXT:    movapd %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2i64_21
; SSE41:         blendpd {{.*}} # xmm1 = xmm1[0],xmm0[1]
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_21_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; SSE2-LABEL: @shuffle_v2i64_21_copy
; SSE2:         shufpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; SSE2-NEXT:    movapd %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @shuffle_v2i64_21_copy
; SSE3:         shufpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; SSE3-NEXT:    movapd %xmm2, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @shuffle_v2i64_21_copy
; SSE41:         blendpd {{.*}} # xmm2 = xmm2[0],xmm1[1]
; SSE41-NEXT:    movapd %xmm2, %xmm0
; SSE41-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 2, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_30(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_30
; ALL:         shufpd {{.*}} # xmm1 = xmm1[1],xmm0[0]
; ALL-NEXT:    movapd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_30_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_30_copy
; ALL:         shufpd {{.*}} # xmm2 = xmm2[1],xmm1[0]
; ALL-NEXT:    movapd %xmm2, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 0>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_31(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_31
; ALL:         punpckhqdq {{.*}} # xmm1 = xmm1[1],xmm0[1]
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 1>
  ret <2 x i64> %shuffle
}
define <2 x i64> @shuffle_v2i64_31_copy(<2 x i64> %nonce, <2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: @shuffle_v2i64_31_copy
; ALL:         punpckhqdq {{.*}} # xmm2 = xmm2[1],xmm1[1]
; ALL-NEXT:    movdqa %xmm2, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 3, i32 1>
  ret <2 x i64> %shuffle
}


define <2 x double> @insert_dup_reg_v2f64(double %a) {
; SSE2-LABEL: @insert_dup_reg_v2f64
; SSE2:         movlhps {{.*}} # xmm0 = xmm0[0,0]
; SSE2-NEXT:    retq
;
; FIXME: This should match movddup as well!
; SSE3-LABEL: @insert_dup_reg_v2f64
; SSE3:         unpcklpd {{.*}} # xmm0 = xmm0[0,0]
; SSE3-NEXT:    retq
;
; FIXME: This should match movddup as well!
; SSE41-LABEL: @insert_dup_reg_v2f64
; SSE41:         unpcklpd {{.*}} # xmm0 = xmm0[0,0]
; SSE41-NEXT:    retq
  %v = insertelement <2 x double> undef, double %a, i32 0
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
}
define <2 x double> @insert_dup_mem_v2f64(double* %ptr) {
; SSE2-LABEL: @insert_dup_mem_v2f64
; SSE2:         movsd {{.*}}, %xmm0
; SSE2-NEXT:    movlhps {{.*}} # xmm0 = xmm0[0,0]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: @insert_dup_mem_v2f64
; SSE3:         movddup {{.*}}, %xmm0
; SSE3-NEXT:    retq
;
; SSE41-LABEL: @insert_dup_mem_v2f64
; SSE41:         movddup {{.*}}, %xmm0
; SSE41-NEXT:    retq
  %a = load double* %ptr
  %v = insertelement <2 x double> undef, double %a, i32 0
  %shuffle = shufflevector <2 x double> %v, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %shuffle
}
