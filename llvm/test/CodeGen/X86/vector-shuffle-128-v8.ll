; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=CHECK-SSE2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <8 x i16> @shuffle_v8i16_01012323(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_01012323
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,0,1,1]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 2, i32 3, i32 2, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_67452301(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_67452301
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,2,1,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 6, i32 7, i32 4, i32 5, i32 2, i32 3, i32 0, i32 1>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_456789AB(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_456789AB
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2:         shufpd {{.*}} # xmm0 = xmm0[1],xmm1[0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_00000000(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_00000000
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_00004444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_00004444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_31206745(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_31206745
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,1,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,3,2]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 6, i32 7, i32 4, i32 5>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44440000(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_44440000
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_75643120(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_75643120
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,0,1]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,1,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,6,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 7, i32 5, i32 6, i32 4, i32 3, i32 1, i32 2, i32 0>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_10545410(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_10545410
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,0,3,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,4,7,6]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 0, i32 5, i32 4, i32 5, i32 4, i32 1, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_54105410(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_54105410
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,4,7,6]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 5, i32 4, i32 1, i32 0, i32 5, i32 4, i32 1, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_54101054(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_54101054
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 5, i32 4, i32 1, i32 0, i32 1, i32 0, i32 5, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04400440(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_04400440
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,4,4,6]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_40044004(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_40044004
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,0,0,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 0, i32 0, i32 4, i32 4, i32 0, i32 0, i32 4>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_26405173(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_26405173
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,4,6]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,1]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,3,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,6,4,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 6, i32 4, i32 0, i32 5, i32 1, i32 7, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_20645173(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_20645173
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,4,6]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,1]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,0,3,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,5,6,4,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 0, i32 6, i32 4, i32 5, i32 1, i32 7, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_26401375(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_26401375
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,5,4,6]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,1,2]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,3,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 2, i32 6, i32 4, i32 0, i32 1, i32 3, i32 7, i32 5>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_00444444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_00444444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,0,2,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 0, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44004444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_44004444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,2,0,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04404444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_04404444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04400000(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_04400000
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,0,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_04404567(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_04404567
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 4, i32 4, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0X444444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_0X444444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,1,2,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 undef, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_44X04444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_44X04444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 4, i32 undef, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_X4404444(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_X4404444
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,2,0,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,4,4,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 4, i32 4, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0127XXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_0127XXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XXXX4563(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_XXXX4563
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,0]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 4, i32 5, i32 6, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_4563XXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_4563XXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,0,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_01274563(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_01274563
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,1,3]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,5,4,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,1,2]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 7, i32 4, i32 5, i32 6, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_45630127(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_45630127
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,1,2,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,0,1,3]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,6,7,5,4]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 5, i32 6, i32 3, i32 0, i32 1, i32 2, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_08192a3b(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_08192a3b
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0c1d2e3f(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_0c1d2e3f
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 12, i32 1, i32 13, i32 2, i32 14, i32 3, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_4c5d6e7f(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_4c5d6e7f
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_48596a7b(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_48596a7b
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 4, i32 8, i32 5, i32 9, i32 6, i32 10, i32 7, i32 11>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_08196e7f(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_08196e7f
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[0,3,2,3]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0c1d6879(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_0c1d6879
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,0,2,3]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 12, i32 1, i32 13, i32 6, i32 8, i32 7, i32 9>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_109832ba(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_109832ba
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm0[2,0,3,1,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,3,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[2,0,3,1,4,5,6,7]
; CHECK-SSE2-NEXT:    punpcklqdq %xmm0, %xmm1
; CHECK-SSE2-NEXT:    movdqa %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 1, i32 0, i32 9, i32 8, i32 3, i32 2, i32 11, i32 10>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_8091a2b3(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_8091a2b3
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    punpcklwd %xmm0, %xmm1
; CHECK-SSE2-NEXT:    movdqa %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 8, i32 0, i32 9, i32 1, i32 10, i32 2, i32 11, i32 3>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_c4d5e6f7(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_c4d5e6f7
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm2 = xmm0[2,3,2,3]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm2, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 12, i32 4, i32 13, i32 5, i32 14, i32 6, i32 15, i32 7>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_0213cedf(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_0213cedf
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,2,1,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,3,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[0,2,1,3,4,5,6,7]
; CHECK-SSE2-NEXT:    punpcklqdq %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 2, i32 1, i32 3, i32 12, i32 14, i32 13, i32 15>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_032dXXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_032dXXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,3,2,1,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 3, i32 2, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}
define <8 x i16> @shuffle_v8i16_XXXcXXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_XXXcXXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm1[2,1,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[0,1,2,1,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_012dXXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_012dXXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[2,1,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm1, %xmm0
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,0,3,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_XXXXcde3(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_XXXXcde3
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; CHECK-SSE2-NEXT:    punpckhwd %xmm0, %xmm1
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm1[0,2,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,0,2]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 12, i32 13, i32 14, i32 3>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_cde3XXXX(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_cde3XXXX
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,1,2,1]
; CHECK-SSE2-NEXT:    punpckhwd %xmm0, %xmm1
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm1[0,2,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,7,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[0,2,2,3]
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 12, i32 13, i32 14, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <8 x i16> %shuffle
}

define <8 x i16> @shuffle_v8i16_012dcde3(<8 x i16> %a, <8 x i16> %b) {
; CHECK-SSE2-LABEL: @shuffle_v8i16_012dcde3
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm2 = xmm0[0,1,2,1]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm3 = xmm1[2,1,2,3]
; CHECK-SSE2-NEXT:    punpckhwd %xmm2, %xmm1
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm1 = xmm1[0,2,2,3,4,5,6,7]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm1 = xmm1[0,1,2,3,4,7,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm1 = xmm1[0,2,2,3]
; CHECK-SSE2-NEXT:    punpcklwd %xmm3, %xmm0
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[3,1,2,0]
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,4,6,6,7]
; CHECK-SSE2-NEXT:    pshufd {{.*}} # xmm0 = xmm0[2,1,2,3]
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[1,2,0,3,4,5,6,7]
; CHECK-SSE2-NEXT:    punpcklqdq %xmm1, %xmm0
; CHECK-SSE2-NEXT:    retq
  %shuffle = shufflevector <8 x i16> %a, <8 x i16> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 12, i32 13, i32 14, i32 3>
  ret <8 x i16> %shuffle
}
