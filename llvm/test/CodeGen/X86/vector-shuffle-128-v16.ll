; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=CHECK-SSE2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <16 x i8> @shuffle_v16i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_00_00_00_00_00_00_00_01_01_01_01_01_01_01_01(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_00_00_00_00_00_00_01_01_01_01_01_01_01_01
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,5,5,5]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_00_00_00_00_00_00_08_08_08_08_08_08_08_08
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,2,4,5,6,7]
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,6,6,6]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_00_00_00_01_01_01_01_02_02_02_02_03_03_03_03(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_00_00_01_01_01_01_02_02_02_02_03_03_03_03
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    punpcklwd %xmm0, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_04_04_04_04_05_05_05_05_06_06_06_06_07_07_07_07(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_04_04_04_04_05_05_05_05_06_06_06_06_07_07_07_07
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    punpckhwd %xmm0, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 4, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7, i32 7>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_00_00_04_04_04_04_08_08_08_08_12_12_12_12
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,2,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,6,6]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 8, i32 8, i32 8, i32 8, i32 12, i32 12, i32 12, i32 12>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_00_01_01_02_02_03_03_04_04_05_05_06_06_07_07(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_00_01_01_02_02_03_03_04_04_05_05_06_06_07_07
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_0101010101010101(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_0101010101010101
; CHECK-SSE2:         pshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_00_16_01_17_02_18_03_19_04_20_05_21_06_22_07_23(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_00_16_01_17_02_18_03_19_04_20_05_21_06_22_07_23
; CHECK-SSE2:         punpcklbw %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_16_00_16_01_16_02_16_03_16_04_16_05_16_06_16_07(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_16_00_16_01_16_02_16_03_16_04_16_05_16_06_16_07
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklbw %xmm1, %xmm1
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    punpcklbw %xmm0, %xmm1
; CHECK-SSE2-NEXT:    movdqa %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 16, i32 0, i32 16, i32 1, i32 16, i32 2, i32 16, i32 3, i32 16, i32 4, i32 16, i32 5, i32 16, i32 6, i32 16, i32 7>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_03_02_01_00_07_06_05_04_11_10_09_08_15_14_13_12(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_03_02_01_00_07_06_05_04_11_10_09_08_15_14_13_12
; CHECK-SSE2:         pxor %xmm1, %xmm1
; CHECK-SSE2-NEXT:    movdqa %xmm0, %xmm2
; CHECK-SSE2-NEXT:    punpckhbw %xmm1, %xmm2
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm2 = xmm2[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm2 = xmm2[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    punpcklbw %xmm1, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    packuswb %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_03_02_01_00_07_06_05_04_19_18_17_16_23_22_21_20(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_03_02_01_00_07_06_05_04_19_18_17_16_23_22_21_20
; CHECK-SSE2:         pxor %xmm2, %xmm2
; CHECK-SSE2-NEXT:    punpcklbw %xmm2, %xmm1
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    punpcklbw %xmm2, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    packuswb %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 19, i32 18, i32 17, i32 16, i32 23, i32 22, i32 21, i32 20>
  ret <16 x i8> %shuffle
}

define <16 x i8> @shuffle_v16i8_03_02_01_00_31_30_29_28_11_10_09_08_23_22_21_20(<16 x i8> %a, <16 x i8> %b) {
; CHECK-SSE2-LABEL: @shuffle_v16i8_03_02_01_00_31_30_29_28_11_10_09_08_23_22_21_20
; CHECK-SSE2:         pxor %xmm2, %xmm2
; CHECK-SSE2-NEXT:    movdqa %xmm1, %xmm3
; CHECK-SSE2-NEXT:    punpcklbw %xmm2, %xmm3
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm3 = xmm3[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    movdqa %xmm0, %xmm4
; CHECK-SSE2-NEXT:    punpckhbw %xmm2, %xmm4
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm4 = xmm4[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    shufpd {{.*}} # xmm4 = xmm4[0],xmm3[1]
; CHECK-SSE2-NEXT:    punpckhbw %xmm2, %xmm1
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    punpcklbw %xmm2, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    shufpd {{.*}} # xmm0 = xmm0[0],xmm1[1]
; CHECK-SSE2-NEXT:    packuswb %xmm4, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> %b, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 31, i32 30, i32 29, i32 28, i32 11, i32 10, i32 9, i32 8, i32 23, i32 22, i32 21, i32 20>
  ret <16 x i8> %shuffle
}

define <16 x i8> @zext_to_v8i16_shuffle(<16 x i8> %a) {
; CHECK-SSE2-LABEL: @zext_to_v8i16_shuffle
; CHECK-SSE2:         pxor %xmm1, %xmm1
; CHECK-SSE2-NEXT:    punpcklbw %xmm1, %xmm0
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 17, i32 1, i32 19, i32 2, i32 21, i32 3, i32 23, i32 4, i32 25, i32 5, i32 27, i32 6, i32 29, i32 7, i32 31>
  ret <16 x i8> %shuffle
}

define <16 x i8> @zext_to_v4i32_shuffle(<16 x i8> %a) {
; CHECK-SSE2-LABEL: @zext_to_v4i32_shuffle
; CHECK-SSE2:         pxor %xmm1, %xmm1
; CHECK-SSE2-NEXT:    punpcklbw %xmm1, %xmm0
; CHECK-SSE2-NEXT:    punpcklbw %xmm1, %xmm0
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 17, i32 18, i32 19, i32 1, i32 21, i32 22, i32 23, i32 2, i32 25, i32 26, i32 27, i32 3, i32 29, i32 30, i32 31>
  ret <16 x i8> %shuffle
}

define <16 x i8> @trunc_v4i32_shuffle(<16 x i8> %a) {
; CHECK-SSE2-LABEL: @trunc_v4i32_shuffle
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pand
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    packuswb %xmm0, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %shuffle
}
