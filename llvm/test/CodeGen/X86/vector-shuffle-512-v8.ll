; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512f | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx512bw | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512BW

target triple = "x86_64-unknown-unknown"

define <8 x double> @shuffle_v8f64_00000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000000:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd %xmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000010(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000010:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00000200(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00000200:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,2,0,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00003000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00003000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,3,0,0,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00040000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00040000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,4,0,0,0,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00500000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00500000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,5,0,0,0,0,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_06000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_06000000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,6,0,0,0,0,0,0]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_70000000(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_70000000:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; ALL-NEXT:    movl $7, %eax
; ALL-NEXT:    vpinsrq $0, %rax, %xmm1, %xmm2
; ALL-NEXT:    vinserti32x4 $0, %xmm2, %zmm1, %zmm1
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01014545(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01014545:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,0,1,4,5,4,5]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00112233(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00112233:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,1,2,2,3,3]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00001111(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00001111:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,1,1,1]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_81a3c5e7(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_81a3c5e7:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,2,11,4,13,6,15]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08080808(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08080808:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,0,8,0,8]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08084c4c(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08084c4c:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,4,12,4,12]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_8823cc67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_8823cc67:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,10,11,4,4,14,15]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9832dc76(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_9832dc76:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,11,10,5,4,15,14]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_9810dc54(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_9810dc54:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,9,8,5,4,13,12]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08194c5d(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08194c5d:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,4,12,5,13]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_2a3b6e7f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_2a3b6e7f:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,10,3,11,6,14,7,15]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08192a3b(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08192a3b:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,2,10,3,11]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_08991abb(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_08991abb:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,1,9,2,3,3]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_091b2d3f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_091b2d3f:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,1,11,2,13,3,15]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_09ab1def(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_09ab1def:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,1,2,3,9,5,6,7]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00014445(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00014445:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,4,4,4,5]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204464(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00204464:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,4,6,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_03004744:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,7,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10005444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10005444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,5,4,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_22006644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,6,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307774(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_33307774:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,7,7,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32107654(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_32107654:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,7,6,5,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00234467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00234467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,4,4,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00224466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00224466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,4,4,6,6]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10325476(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10325476:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,5,4,7,6]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11335577(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_11335577:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,5,5,7,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10235467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,3,5,4,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10225466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10225466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,2,5,4,6,6]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00015444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00015444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,5,4,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00204644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00204644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,6,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_03004474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_03004474:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,4,7,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10004444(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10004444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,4,4,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_22006446(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_22006446:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,4,4,6]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_33307474(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_33307474:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,4,7,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_32104567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_32104567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,4,5,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00236744(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00236744:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,6,7,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00226644(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00226644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,6,6,4,4]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_10324567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_10324567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,4,5,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_11334567(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_11334567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,4,5,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235467(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01235467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,7]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_01235466(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_01235466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,6]
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_002u6u44(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_002u6u44:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,2,u,6,u,4,4>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_00uu66uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_00uu66uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,6,6,u,u>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_103245uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_103245uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,3,2,4,5,u,u>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_1133uu67(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_1133uu67:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,1,3,3,u,u,6,7>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_0uu354uu(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_0uu354uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,u,u,3,5,4,u,u>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_uuu3uu66(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_uuu3uu66:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,3,u,u,6,6>
; ALL-NEXT:    vpermpd %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_c348cda0(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_c348cda0:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [4,11,12,0,4,5,2,8]
; ALL-NEXT:    vpermt2pd %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 12, i32 3, i32 4, i32 8, i32 12, i32 13, i32 10, i32 0>
  ret <8 x double> %shuffle
}

define <8 x double> @shuffle_v8f64_f511235a(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_f511235a:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [15,5,1,1,2,3,5,10]
; ALL-NEXT:    vpermt2pd %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32> <i32 15, i32 5, i32 1, i32 1, i32 2, i32 3, i32 5, i32 10>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000000:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq %xmm0, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000010(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000010:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,0,1,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00000200(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00000200:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,0,2,0,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00003000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00003000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,3,0,0,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 3, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00040000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00040000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,4,0,0,0,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00500000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00500000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,5,0,0,0,0,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 5, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_06000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_06000000:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,6,0,0,0,0,0,0]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_70000000(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_70000000:
; ALL:       # BB#0:
; ALL-NEXT:    vpxord %zmm1, %zmm1, %zmm1
; ALL-NEXT:    movl $7, %eax
; ALL-NEXT:    vpinsrq $0, %rax, %xmm1, %xmm2
; ALL-NEXT:    vinserti32x4 $0, %xmm2, %zmm1, %zmm1
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01014545(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01014545:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,0,1,4,5,4,5]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 4, i32 5, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00112233(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00112233:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,1,1,2,2,3,3]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00001111(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00001111:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,0,1,1,1,1]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_81a3c5e7(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_81a3c5e7:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,2,11,4,13,6,15]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08080808(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08080808:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,0,8,0,8]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 0, i32 8, i32 0, i32 8>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08084c4c(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08084c4c:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,0,8,4,12,4,12]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 0, i32 8, i32 4, i32 12, i32 4, i32 12>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_8823cc67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_8823cc67:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,0,10,11,4,4,14,15]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 8, i32 8, i32 2, i32 3, i32 12, i32 12, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9832dc76(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_9832dc76:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,11,10,5,4,15,14]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 3, i32 2, i32 13, i32 12, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_9810dc54(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_9810dc54:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [1,0,9,8,5,4,13,12]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 9, i32 8, i32 1, i32 0, i32 13, i32 12, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08194c5d(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08194c5d:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,4,12,5,13]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_2a3b6e7f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_2a3b6e7f:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [2,10,3,11,6,14,7,15]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08192a3b(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08192a3b:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,8,1,9,2,10,3,11]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_08991abb(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_08991abb:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,0,1,1,9,2,3,3]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 8, i32 9, i32 9, i32 1, i32 10, i32 11, i32 11>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_091b2d3f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_091b2d3f:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [0,9,1,11,2,13,3,15]
; ALL-NEXT:    vpermt2q %zmm1, %zmm2, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 1, i32 11, i32 2, i32 13, i32 3, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_09ab1def(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_09ab1def:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [8,1,2,3,9,5,6,7]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 1, i32 13, i32 14, i32 15>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00014445(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00014445:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,4,4,4,5]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 4, i32 4, i32 4, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204464(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00204464:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,4,6,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 4, i32 6, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_03004744:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,7,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10005444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10005444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,5,4,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_22006644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,6,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307774(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_33307774:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,7,7,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 7, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32107654(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_32107654:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,7,6,5,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00234467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00234467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,4,4,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 4, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00224466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00224466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,4,4,6,6]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10325476(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10325476:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,5,4,7,6]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11335577(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_11335577:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,5,5,7,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10235467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,3,5,4,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10225466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10225466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,2,2,5,4,6,6]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 2, i32 2, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00015444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00015444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,0,1,5,4,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 0, i32 1, i32 5, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00204644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00204644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,0,4,6,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 0, i32 4, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_03004474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_03004474:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,3,0,0,4,4,7,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 3, i32 0, i32 0, i32 4, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10004444(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10004444:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,0,0,4,4,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_22006446(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_22006446:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [2,2,0,0,6,4,4,6]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 2, i32 2, i32 0, i32 0, i32 6, i32 4, i32 4, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_33307474(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_33307474:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,3,3,0,7,4,7,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 3, i32 3, i32 0, i32 7, i32 4, i32 7, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_32104567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_32104567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [3,2,1,0,4,5,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00236744(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00236744:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,3,6,7,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 3, i32 6, i32 7, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00226644(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00226644:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,0,2,2,6,6,4,4]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 6, i32 6, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_10324567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_10324567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,0,3,2,4,5,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_11334567(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_11334567:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [1,1,3,3,4,5,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235467(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01235467:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,7]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_01235466(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_01235466:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = [0,1,2,3,5,4,6,6]
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 4, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_002u6u44(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_002u6u44:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,2,u,6,u,4,4>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 4, i32 4>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_00uu66uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_00uu66uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,0,u,u,6,6,u,u>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 0, i32 undef, i32 undef, i32 6, i32 6, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_103245uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_103245uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,0,3,2,4,5,u,u>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 4, i32 5, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_1133uu67(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_1133uu67:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <1,1,3,3,u,u,6,7>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 undef, i32 undef, i32 6, i32 7>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_0uu354uu(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_0uu354uu:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <0,u,u,3,5,4,u,u>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 0, i32 undef, i32 undef, i32 3, i32 5, i32 4, i32 undef, i32 undef>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_uuu3uu66(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_uuu3uu66:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm1 = <u,u,u,3,u,u,6,6>
; ALL-NEXT:    vpermq %zmm0, %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 3, i32 undef, i32 undef, i32 6, i32 6>
  ret <8 x i64> %shuffle
}

define <8 x i64> @shuffle_v8i64_6caa87e5(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_6caa87e5:
; ALL:       # BB#0:
; ALL-NEXT:    vmovdqa64 {{.*#+}} zmm2 = [14,4,2,2,0,15,6,13]
; ALL-NEXT:    vpermt2q %zmm0, %zmm2, %zmm1
; ALL-NEXT:    vmovaps %zmm1, %zmm0
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32> <i32 6, i32 12, i32 10, i32 10, i32 8, i32 7, i32 14, i32 5>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_082a4c6e(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_082a4c6e:
; ALL:       # BB#0:
; ALL-NEXT:    vunpcklpd {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_082a4c6e(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_082a4c6e:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpcklqdq {{.*#+}} zmm0 = zmm0[0],zmm1[0],zmm0[2],zmm1[2],zmm0[4],zmm1[4],zmm0[6],zmm1[6]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  ret <8 x i64> %shuffle
}

define <8 x double> @shuffle_v8f64_193b5d7f(<8 x double> %a, <8 x double> %b) {
; ALL-LABEL: shuffle_v8f64_193b5d7f:
; ALL:       # BB#0:
; ALL-NEXT:    vunpckhpd {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x double> %a, <8 x double> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x double> %shuffle
}

define <8 x i64> @shuffle_v8i64_193b5d7f(<8 x i64> %a, <8 x i64> %b) {
; ALL-LABEL: shuffle_v8i64_193b5d7f:
; ALL:       # BB#0:
; ALL-NEXT:    vpunpckhqdq {{.*#+}} zmm0 = zmm0[1],zmm1[1],zmm0[3],zmm1[3],zmm0[5],zmm1[5],zmm0[7],zmm1[7]
; ALL-NEXT:    retq
  %shuffle = shufflevector <8 x i64> %a, <8 x i64> %b, <8 x i32><i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <8 x i64> %shuffle
}
