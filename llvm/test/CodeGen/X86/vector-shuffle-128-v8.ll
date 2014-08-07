; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=ALL --check-prefix=SSSE3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <8 x i16> @shuffle_v8i16_01012323(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_01012323
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,0,1,1]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 2, i32 3, i32 2, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_67452301(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_67452301
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,2,1,0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 6, i32 7, i32 4, i32 5, i32 2, i32 3, i32 0, i32 1>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_456789AB(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_456789AB
; ALL:       # BB#0:
; ALL:         shufpd {{.*}} # xmm0 = xmm0[1],xmm1[0]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_00000000(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_00000000
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_00000000
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_00004444(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_00004444
; ALL:       # BB#0:
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; ALL-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_31206745(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_31206745
; ALL:       # BB#0:
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,1,2,0,4,5,6,7]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,3,2]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 6, i32 7, i32 4, i32 5>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44440000(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_44440000
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,0,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_44440000
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,8,9,8,9,8,9,0,1,0,1,0,1,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_75643120(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_75643120
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,1,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,6,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_75643120
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[14,15,10,11,12,13,8,9,6,7,2,3,4,5,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 7, i32 5, i32 6, i32 4, i32 3, i32 1, i32 2, i32 0>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_10545410(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_10545410
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,0,3,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,4,7,6]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_10545410
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[2,3,0,1,10,11,8,9,10,11,8,9,2,3,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 0, i32 5, i32 4, i32 5, i32 4, i32 1, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_54105410(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_54105410
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,4,7,6]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_54105410
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[10,11,8,9,2,3,0,1,10,11,8,9,2,3,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 5, i32 4, i32 1, i32 0, i32 5, i32 4, i32 1, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_54101054(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_54101054
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_54101054
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[10,11,8,9,2,3,0,1,2,3,0,1,10,11,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 5, i32 4, i32 1, i32 0, i32 1, i32 0, i32 5, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04400440(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_04400440
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,4,4,6]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_04400440
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,8,9,8,9,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_40044004(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_40044004
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,0,0,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_40044004
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,0,1,0,1,8,9,8,9,0,1,0,1,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 0, i32 0, i32 4, i32 4, i32 0, i32 0, i32 4>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_26405173(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_26405173
; SSE2:       # BB#0:
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,6,4]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,1]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,3,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,6,4,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_26405173
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[4,5,12,13,8,9,0,1,10,11,2,3,14,15,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 6, i32 4, i32 0, i32 5, i32 1, i32 7, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_20645173(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_20645173
; SSE2:       # BB#0:
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,6,4]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,1]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,0,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,6,4,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_20645173
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[4,5,0,1,12,13,8,9,10,11,2,3,14,15,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 0, i32 6, i32 4, i32 5, i32 1, i32 7, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_26401375(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_26401375
; SSE2:       # BB#0:
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,6,4]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,1,2]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,3,0,4,5,6,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_26401375
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[4,5,12,13,8,9,0,1,2,3,6,7,14,15,10,11]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 6, i32 4, i32 0, i32 1, i32 3, i32 7, i32 5>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_66751643(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_66751643
; SSE2:       # BB#0:
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,5,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,1,3,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,4,6]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_66751643
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[12,13,12,13,14,15,10,11,2,3,12,13,8,9,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 6, i32 6, i32 7, i32 5, i32 1, i32 6, i32 4, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_60514754(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_60514754
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,5,4,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,0,3,1,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,7,5,6]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_60514754
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[12,13,0,1,10,11,2,3,8,9,14,15,10,11,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 6, i32 0, i32 5, i32 1, i32 4, i32 7, i32 5, i32 4>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_00444444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_00444444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,2,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_00444444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,0,1,8,9,8,9,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44004444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_44004444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,2,0,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_44004444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,8,9,0,1,0,1,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04404444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_04404444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_04404444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04400000(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_04400000
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,0,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_04400000
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,8,9,8,9,0,1,0,1,0,1,0,1,0,1]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04404567(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_04404567
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0X444444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_0X444444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,1,2,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_0X444444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,{{[0-9]+,[0-9]+}},8,9,8,9,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 undef, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44X04444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_44X04444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,2,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_44X04444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,8,9,{{[0-9]+,[0-9]+}},0,1,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 undef, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_X4404444(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_X4404444
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_X4404444
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[{{[0-9]+,[0-9]+}},8,9,8,9,0,1,8,9,8,9,8,9,8,9]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0127XXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_0127XXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_0127XXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,2,3,4,5,14,15,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XXXX4563(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_XXXX4563
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_XXXX4563
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}},8,9,10,11,12,13,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 4, i32 5, i32 6, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_4563XXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_4563XXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,0,2,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_4563XXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,10,11,12,13,6,7,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_01274563(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_01274563
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,5,4,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,1,2]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_01274563
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,2,3,4,5,14,15,8,9,10,11,12,13,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 7, i32 4, i32 5, i32 6, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_45630127(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_45630127
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,1,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,0,3,1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_45630127
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[8,9,10,11,12,13,6,7,0,1,2,3,4,5,14,15]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 3, i32 0, i32 1, i32 2, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_08192a3b(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_08192a3b
; ALL:       # BB#0:
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0c1d2e3f(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_0c1d2e3f
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 12, i32 1, i32 13, i32 2, i32 14, i32 3, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_4c5d6e7f(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_4c5d6e7f
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_48596a7b(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_48596a7b
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 8, i32 5, i32 9, i32 6, i32 10, i32 7, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_08196e7f(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_08196e7f
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,3]
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0c1d6879(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_0c1d6879
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,0,2,3]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,3]
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 12, i32 1, i32 13, i32 6, i32 8, i32 7, i32 9>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_109832ba(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_109832ba
; ALL:       # BB#0:
; ALL-NEXT:    punpcklwd %xmm1, %xmm0
; ALL-NEXT:    pshuflw {{.*}} # xmm1 = xmm0[2,0,3,1,4,5,6,7]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,0,3,1,4,5,6,7]
; ALL-NEXT:    punpcklqdq %xmm0, %xmm1
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 0, i32 9, i32 8, i32 3, i32 2, i32 11, i32 10>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_8091a2b3(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_8091a2b3
; ALL:       # BB#0:
; ALL-NEXT:    punpcklwd %xmm0, %xmm1
; ALL-NEXT:    movdqa %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 0, i32 9, i32 1, i32 10, i32 2, i32 11, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_c4d5e6f7(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_c4d5e6f7
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm2 = xmm0[2,3,2,3]
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm1[2,3,2,3]
; ALL-NEXT:    punpcklwd %xmm2, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 12, i32 4, i32 13, i32 5, i32 14, i32 6, i32 15, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0213cedf(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_0213cedf
; ALL:       # BB#0:
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; ALL-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; ALL-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[0,2,1,3,4,5,6,7]
; ALL-NEXT:    punpcklqdq %xmm1, %xmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 1, i32 3, i32 12, i32 14, i32 13, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_443aXXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_443aXXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}}          # xmm0 = xmm0[2,1,2,3]
; SSE2-NEXT:    pshuflw {{.*}}         # xmm0 = xmm0[0,0,2,3,4,5,6,7]
; SSE2-NEXT:    punpcklwd %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE2-NEXT:    pshuflw {{.*}}         # xmm0 = xmm0[0,2,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}}         # xmm0 = xmm0[0,1,2,3,6,5,6,7]
; SSE2-NEXT:    pshufd {{.*}}          # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_443aXXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}}          # xmm0 = xmm0[2,1,2,3]
; SSSE3-NEXT:    pshuflw {{.*}}         # xmm0 = xmm0[0,0,2,3,4,5,6,7]
; SSSE3-NEXT:    punpcklwd %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,4,5,12,13,10,11,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 3, i32 10, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_032dXXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_032dXXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,1,4,5,6,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_032dXXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; SSSE3-NEXT:    punpcklwd %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,12,13,8,9,6,7,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 3, i32 2, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_XXXcXXXX(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: @shuffle_v8i16_XXXcXXXX
; ALL:       # BB#0:
; ALL-NEXT:    pshufd {{.*}} # xmm0 = xmm1[2,1,2,3]
; ALL-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,1,2,1,4,5,6,7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_012dXXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_012dXXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,0,3,4,5,6,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_012dXXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; SSSE3-NEXT:    punpcklwd %xmm1, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,4,5,8,9,6,7,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XXXXcde3(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_XXXXcde3
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; SSE2-NEXT:    punpckhwd %xmm0, %xmm1
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm1[0,2,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,2]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_XXXXcde3
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; SSSE3-NEXT:    punpckhwd %xmm0, %xmm1 # xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    pshufb {{.*}} # xmm1 = xmm1[{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}},0,1,4,5,8,9,14,15]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_cde3XXXX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_cde3XXXX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; SSE2-NEXT:    punpckhwd %xmm0, %xmm1
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm1[0,2,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_cde3XXXX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; SSSE3-NEXT:    punpckhwd %xmm0, %xmm1 # xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    pshufb {{.*}} # xmm1 = xmm1[0,1,4,5,8,9,14,15,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 12, i32 13, i32 14, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_012dcde3(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_012dcde3
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}} # xmm2 = xmm0[0,1,2,1]
; SSE2-NEXT:    pshufd {{.*}} # xmm3 = xmm1[2,1,2,3]
; SSE2-NEXT:    punpckhwd %xmm2, %xmm1
; SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[0,2,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,7,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; SSE2-NEXT:    punpcklwd %xmm3, %xmm0
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,2,3]
; SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,0,3,4,5,6,7]
; SSE2-NEXT:    punpcklqdq %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_012dcde3
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}} # xmm2 = xmm0[0,1,2,1]
; SSSE3-NEXT:    pshufd {{.*}} # xmm3 = xmm1[2,1,2,3]
; SSSE3-NEXT:    punpckhwd %xmm2, %xmm1 # xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSSE3-NEXT:    pshufb {{.*}} # xmm1 = xmm1[0,1,4,5,8,9,14,15,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    punpcklwd %xmm3, %xmm0 # xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[0,1,4,5,8,9,6,7,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    punpcklqdq %xmm1, %xmm0
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 12, i32 13, i32 14, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XXX1X579(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_XXX1X579
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufhw {{.*}}   # xmm0 = xmm0[0,1,2,3,7,5,6,7]
; SSE2-NEXT:    pshufd {{.*}}    # xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*}}   # xmm0 = xmm0[0,1,3,2,4,5,6,7]
; SSE2-NEXT:    punpcklwd {{.*}} # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE2-NEXT:    pshufhw {{.*}}   # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; SSE2-NEXT:    pshufd {{.*}}    # xmm0 = xmm0[0,1,2,1]
; SSE2-NEXT:    pshuflw {{.*}}   # xmm0 = xmm0[0,1,2,2,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}}   # xmm0 = xmm0[0,1,2,3,4,4,5,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_XXX1X579
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufb {{.*}}    # xmm0 = xmm0[{{[0-9]+,[0-9]+}},2,3,10,11,14,15,{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    punpcklwd {{.*}} # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    pshufb {{.*}}    # xmm0 = xmm0[{{[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+}},4,5,{{[0-9]+,[0-9]+}},8,9,12,13,6,7]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 1, i32 undef, i32 5, i32 7, i32 9>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XX4X8acX(<8 x i16> %a, <8 x i16> %b) {
; SSE2-LABEL: @shuffle_v8i16_XX4X8acX
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*}}    # xmm0 = xmm0[2,1,2,3]
; SSE2-NEXT:    pshuflw {{.*}}   # xmm1 = xmm1[0,2,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*}}    # xmm1 = xmm1[0,2,2,3]
; SSE2-NEXT:    punpcklwd {{.*}} # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE2-NEXT:    pshuflw {{.*}}   # xmm0 = xmm1[0,1,2,0,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*}}    # xmm0 = xmm0[0,1,2,1]
; SSE2-NEXT:    pshuflw {{.*}}   # xmm0 = xmm0[0,1,1,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*}}   # xmm0 = xmm0[0,1,2,3,7,6,4,7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: @shuffle_v8i16_XX4X8acX
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*}}    # [[X:xmm[0-9]+]] = xmm0[2,1,2,3]
; SSSE3-NEXT:    pshuflw {{.*}}   # xmm0 = xmm1[0,2,2,3,4,5,6,7]
; SSSE3-NEXT:    pshufd {{.*}}    # xmm0 = xmm0[0,2,2,3]
; SSSE3-NEXT:    punpcklwd {{.*}} # xmm0 = xmm0[0],[[X]][0],xmm0[1],[[X]][1],xmm0[2],[[X]][2],xmm0[3],[[X]][3]
; SSSE3-NEXT:    pshufb {{.*}} # xmm0 = xmm0[{{[0-9]+,[0-9]+,[0-9]+,[0-9]+}},2,3,{{[0-9]+,[0-9]+}},0,1,4,5,8,9,{{[0-9]+,[0-9]+}}]
; SSSE3-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 4, i32 undef, i32 8, i32 10, i32 12, i32 undef>
  ret <8 x i16> %shuffle
}
