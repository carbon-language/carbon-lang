; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

; AVX128 tests:

define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
; SSE2-LABEL: vsel_float:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_float:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_float:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_float:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

define <4 x float> @vsel_float2(<4 x float> %v1, <4 x float> %v2) {
; SSE2-LABEL: vsel_float2:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movss {{.*#+}} xmm1 = xmm0[0],xmm1[1,2,3]
; SSE2-NEXT:    movaps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_float2:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movss {{.*#+}} xmm1 = xmm0[0],xmm1[1,2,3]
; SSSE3-NEXT:    movaps %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_float2:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm1[1,2,3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_float2:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1,2,3]
; AVX-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
; SSE2-LABEL: vsel_4xi8:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm0[3,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,2]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_4xi8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm0[3,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,1],xmm1[0,2]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_4xi8:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_4xi8:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_4xi8:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0,1],xmm1[2],xmm0[3]
; AVX2-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i8> %v1, <4 x i8> %v2
  ret <4 x i8> %vsel
}

define <4 x i16> @vsel_4xi16(<4 x i16> %v1, <4 x i16> %v2) {
; SSE2-LABEL: vsel_4xi16:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,0],xmm0[0,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm0[2,3]
; SSE2-NEXT:    movaps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_4xi16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,0],xmm0[0,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm0[2,3]
; SSSE3-NEXT:    movaps %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_4xi16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_4xi16:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_4xi16:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2,3]
; AVX2-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i16> %v1, <4 x i16> %v2
  ret <4 x i16> %vsel
}

define <4 x i32> @vsel_i32(<4 x i32> %v1, <4 x i32> %v2) {
; SSE2-LABEL: vsel_i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,3,2,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[1,3,2,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,2,3]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX2-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> %v1, <4 x i32> %v2
  ret <4 x i32> %vsel
}

define <2 x double> @vsel_double(<2 x double> %v1, <2 x double> %v2) {
; SSE2-LABEL: vsel_double:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_double:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSSE3-NEXT:    movapd %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_double:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm1[1]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_double:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendpd {{.*#+}} xmm0 = xmm0[0],xmm1[1]
; AVX-NEXT:    retq
entry:
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x double> %v1, <2 x double> %v2
  ret <2 x double> %vsel
}

define <2 x i64> @vsel_i64(<2 x i64> %v1, <2 x i64> %v2) {
; SSE2-LABEL: vsel_i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSE2-NEXT:    movapd %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm1 = xmm0[0],xmm1[1]
; SSSE3-NEXT:    movapd %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5,6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3]
; AVX2-NEXT:    retq
entry:
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x i64> %v1, <2 x i64> %v2
  ret <2 x i64> %vsel
}

define <8 x i16> @vsel_8xi16(<8 x i16> %v1, <8 x i16> %v2) {
; SSE2-LABEL: vsel_8xi16:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movaps {{.*#+}} xmm2 = [0,65535,65535,65535,0,65535,65535,65535]
; SSE2-NEXT:    andps %xmm2, %xmm1
; SSE2-NEXT:    andnps %xmm0, %xmm2
; SSE2-NEXT:    orps %xmm1, %xmm2
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_8xi16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movaps {{.*#+}} xmm2 = [0,65535,65535,65535,0,65535,65535,65535]
; SSSE3-NEXT:    andps %xmm2, %xmm1
; SSSE3-NEXT:    andnps %xmm0, %xmm2
; SSSE3-NEXT:    orps %xmm1, %xmm2
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_8xi16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1,2,3],xmm0[4],xmm1[5,6,7]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_8xi16:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpblendw {{.*#+}} xmm0 = xmm0[0],xmm1[1,2,3],xmm0[4],xmm1[5,6,7]
; AVX-NEXT:    retq
entry:
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i16> %v1, <8 x i16> %v2
  ret <8 x i16> %vsel
}

define <16 x i8> @vsel_i8(<16 x i8> %v1, <16 x i8> %v2) {
; SSE2-LABEL: vsel_i8:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    movdqa %xmm1, %xmm3
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm3 = xmm3[8],xmm2[8],xmm3[9],xmm2[9],xmm3[10],xmm2[10],xmm3[11],xmm2[11],xmm3[12],xmm2[12],xmm3[13],xmm2[13],xmm3[14],xmm2[14],xmm3[15],xmm2[15]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm3 = xmm3[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm3 = xmm3[0,1,2,3,7,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm3 = xmm3[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm3 = xmm3[1,0,3,2,4,5,6,7]
; SSE2-NEXT:    movdqa %xmm1, %xmm4
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm4 = xmm4[0],xmm2[0],xmm4[1],xmm2[1],xmm4[2],xmm2[2],xmm4[3],xmm2[3],xmm4[4],xmm2[4],xmm4[5],xmm2[5],xmm4[6],xmm2[6],xmm4[7],xmm2[7]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm2 = xmm4[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm2 = xmm2[0,1,2,3,7,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm2[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm2 = xmm2[1,0,3,2,4,5,6,7]
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; SSE2-NEXT:    packuswb %xmm0, %xmm2
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE2-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm1 = xmm1[0,1,2,3,7,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[0,2,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm1 = xmm1[1,0,3,2,4,5,6,7]
; SSE2-NEXT:    packuswb %xmm0, %xmm1
; SSE2-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:    packuswb %xmm0, %xmm0
; SSE2-NEXT:    packuswb %xmm0, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm1, %xmm2
; SSSE3-NEXT:    pshufb {{.*#+}} xmm2 = xmm2[2,6,10,14,u,u,u,u,u,u,u,u,u,u,u,u]
; SSSE3-NEXT:    pshufb {{.*#+}} xmm0 = xmm0[0,4,8,12,u,u,u,u,u,u,u,u,u,u,u,u]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[1,3,5,7,9,11,13,15,u,u,u,u,u,u,u,u]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i8:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm2
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0]
; SSE41-NEXT:    pblendvb %xmm2, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_i8:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovdqa {{.*#+}} xmm2 = [255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0]
; AVX-NEXT:    vpblendvb %xmm2, %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %vsel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i8> %v1, <16 x i8> %v2
  ret <16 x i8> %vsel
}


; AVX256 tests:

define <8 x float> @vsel_float8(<8 x float> %v1, <8 x float> %v2) {
; SSE2-LABEL: vsel_float8:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSE2-NEXT:    movss {{.*#+}} xmm3 = xmm1[0],xmm3[1,2,3]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_float8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSSE3-NEXT:    movss {{.*#+}} xmm3 = xmm1[0],xmm3[1,2,3]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_float8:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm2[1,2,3]
; SSE41-NEXT:    blendps {{.*#+}} xmm1 = xmm1[0],xmm3[1,2,3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_float8:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7]
; AVX-NEXT:    retq
entry:
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x float> %v1, <8 x float> %v2
  ret <8 x float> %vsel
}

define <8 x i32> @vsel_i328(<8 x i32> %v1, <8 x i32> %v2) {
; SSE2-LABEL: vsel_i328:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSE2-NEXT:    movss {{.*#+}} xmm3 = xmm1[0],xmm3[1,2,3]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i328:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSSE3-NEXT:    movss {{.*#+}} xmm3 = xmm1[0],xmm3[1,2,3]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i328:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm2[2,3,4,5,6,7]
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm1[0,1],xmm3[2,3,4,5,6,7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_i328:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_i328:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3],ymm0[4],ymm1[5,6,7]
; AVX2-NEXT:    retq
entry:
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i32> %v1, <8 x i32> %v2
  ret <8 x i32> %vsel
}

define <8 x double> @vsel_double8(<8 x double> %v1, <8 x double> %v2) {
; SSE2-LABEL: vsel_double8:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm4 = xmm0[0],xmm4[1]
; SSE2-NEXT:    movsd {{.*#+}} xmm6 = xmm2[0],xmm6[1]
; SSE2-NEXT:    movapd %xmm4, %xmm0
; SSE2-NEXT:    movaps %xmm5, %xmm1
; SSE2-NEXT:    movapd %xmm6, %xmm2
; SSE2-NEXT:    movaps %xmm7, %xmm3
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_double8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm4 = xmm0[0],xmm4[1]
; SSSE3-NEXT:    movsd {{.*#+}} xmm6 = xmm2[0],xmm6[1]
; SSSE3-NEXT:    movapd %xmm4, %xmm0
; SSSE3-NEXT:    movaps %xmm5, %xmm1
; SSSE3-NEXT:    movapd %xmm6, %xmm2
; SSSE3-NEXT:    movaps %xmm7, %xmm3
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_double8:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm4[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm2 = xmm2[0],xmm6[1]
; SSE41-NEXT:    movaps %xmm5, %xmm1
; SSE41-NEXT:    movaps %xmm7, %xmm3
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_double8:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm2[1,2,3]
; AVX-NEXT:    vblendpd {{.*#+}} ymm1 = ymm1[0],ymm3[1,2,3]
; AVX-NEXT:    retq
entry:
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x double> %v1, <8 x double> %v2
  ret <8 x double> %vsel
}

define <8 x i64> @vsel_i648(<8 x i64> %v1, <8 x i64> %v2) {
; SSE2-LABEL: vsel_i648:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm4 = xmm0[0],xmm4[1]
; SSE2-NEXT:    movsd {{.*#+}} xmm6 = xmm2[0],xmm6[1]
; SSE2-NEXT:    movapd %xmm4, %xmm0
; SSE2-NEXT:    movaps %xmm5, %xmm1
; SSE2-NEXT:    movapd %xmm6, %xmm2
; SSE2-NEXT:    movaps %xmm7, %xmm3
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i648:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm4 = xmm0[0],xmm4[1]
; SSSE3-NEXT:    movsd {{.*#+}} xmm6 = xmm2[0],xmm6[1]
; SSSE3-NEXT:    movapd %xmm4, %xmm0
; SSSE3-NEXT:    movaps %xmm5, %xmm1
; SSSE3-NEXT:    movapd %xmm6, %xmm2
; SSSE3-NEXT:    movaps %xmm7, %xmm3
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i648:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm4[4,5,6,7]
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1,2,3],xmm6[4,5,6,7]
; SSE41-NEXT:    movaps %xmm5, %xmm1
; SSE41-NEXT:    movaps %xmm7, %xmm3
; SSE41-NEXT:    retq
;
; AVX1-LABEL: vsel_i648:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm2[1,2,3]
; AVX1-NEXT:    vblendpd {{.*#+}} ymm1 = ymm1[0],ymm3[1,2,3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: vsel_i648:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm0[0,1],ymm2[2,3,4,5,6,7]
; AVX2-NEXT:    vpblendd {{.*#+}} ymm1 = ymm1[0,1],ymm3[2,3,4,5,6,7]
; AVX2-NEXT:    retq
entry:
  %vsel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x i64> %v1, <8 x i64> %v2
  ret <8 x i64> %vsel
}

define <4 x double> @vsel_double4(<4 x double> %v1, <4 x double> %v2) {
; SSE2-LABEL: vsel_double4:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm2 = xmm0[0],xmm2[1]
; SSE2-NEXT:    movsd {{.*#+}} xmm3 = xmm1[0],xmm3[1]
; SSE2-NEXT:    movapd %xmm2, %xmm0
; SSE2-NEXT:    movapd %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_double4:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm2 = xmm0[0],xmm2[1]
; SSSE3-NEXT:    movsd {{.*#+}} xmm3 = xmm1[0],xmm3[1]
; SSSE3-NEXT:    movapd %xmm2, %xmm0
; SSSE3-NEXT:    movapd %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_double4:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm2[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm1 = xmm1[0],xmm3[1]
; SSE41-NEXT:    retq
;
; AVX-LABEL: vsel_double4:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2],ymm1[3]
; AVX-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %v1, <4 x double> %v2
  ret <4 x double> %vsel
}

define <2 x double> @testa(<2 x double> %x, <2 x double> %y) {
; SSE2-LABEL: testa:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movapd %xmm1, %xmm2
; SSE2-NEXT:    cmplepd %xmm0, %xmm2
; SSE2-NEXT:    andpd %xmm2, %xmm0
; SSE2-NEXT:    andnpd %xmm1, %xmm2
; SSE2-NEXT:    orpd %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: testa:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movapd %xmm1, %xmm2
; SSSE3-NEXT:    cmplepd %xmm0, %xmm2
; SSSE3-NEXT:    andpd %xmm2, %xmm0
; SSSE3-NEXT:    andnpd %xmm1, %xmm2
; SSSE3-NEXT:    orpd %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testa:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movapd %xmm0, %xmm2
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    cmplepd %xmm2, %xmm0
; SSE41-NEXT:    blendvpd %xmm2, %xmm1
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testa:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vcmplepd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

define <2 x double> @testb(<2 x double> %x, <2 x double> %y) {
; SSE2-LABEL: testb:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movapd %xmm1, %xmm2
; SSE2-NEXT:    cmpnlepd %xmm0, %xmm2
; SSE2-NEXT:    andpd %xmm2, %xmm0
; SSE2-NEXT:    andnpd %xmm1, %xmm2
; SSE2-NEXT:    orpd %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: testb:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movapd %xmm1, %xmm2
; SSSE3-NEXT:    cmpnlepd %xmm0, %xmm2
; SSSE3-NEXT:    andpd %xmm2, %xmm0
; SSSE3-NEXT:    andnpd %xmm1, %xmm2
; SSSE3-NEXT:    orpd %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testb:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movapd %xmm0, %xmm2
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    cmpnlepd %xmm2, %xmm0
; SSE41-NEXT:    blendvpd %xmm2, %xmm1
; SSE41-NEXT:    movapd %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testb:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vcmpnlepd %xmm0, %xmm1, %xmm2
; AVX-NEXT:    vblendvpd %xmm2, %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %min_is_x = fcmp ult <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; If we can figure out a blend has a constant mask, we should emit the
; blend instruction with an immediate mask
define <4 x double> @constant_blendvpd_avx(<4 x double> %xy, <4 x double> %ab) {
; SSE2-LABEL: constant_blendvpd_avx:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm3 = xmm1[0],xmm3[1]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movapd %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: constant_blendvpd_avx:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm3 = xmm1[0],xmm3[1]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movapd %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: constant_blendvpd_avx:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm1 = xmm1[0],xmm3[1]
; SSE41-NEXT:    movaps %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: constant_blendvpd_avx:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2],ymm1[3]
; AVX-NEXT:    retq
entry:
  %select = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x double> %xy, <4 x double> %ab
  ret <4 x double> %select
}

define <8 x float> @constant_blendvps_avx(<8 x float> %xyzw, <8 x float> %abcd) {
; SSE2-LABEL: constant_blendvps_avx:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,0],xmm2[2,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,1],xmm0[2,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,0],xmm3[2,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[2,0]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: constant_blendvps_avx:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[3,0],xmm2[2,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,1],xmm0[2,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[3,0],xmm3[2,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[2,0]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: constant_blendvps_avx:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm2[0,1,2],xmm0[3]
; SSE41-NEXT:    blendps {{.*#+}} xmm1 = xmm3[0,1,2],xmm1[3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: constant_blendvps_avx:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} ymm0 = ymm1[0,1,2],ymm0[3],ymm1[4,5,6],ymm0[7]
; AVX-NEXT:    retq
entry:
  %select = select <8 x i1> <i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true>, <8 x float> %xyzw, <8 x float> %abcd
  ret <8 x float> %select
}

define <32 x i8> @constant_pblendvb_avx2(<32 x i8> %xyzw, <32 x i8> %abcd) {
; SSE2-LABEL: constant_pblendvb_avx2:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pxor %xmm8, %xmm8
; SSE2-NEXT:    movdqa %xmm0, %xmm4
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm4 = xmm4[8],xmm8[8],xmm4[9],xmm8[9],xmm4[10],xmm8[10],xmm4[11],xmm8[11],xmm4[12],xmm8[12],xmm4[13],xmm8[13],xmm4[14],xmm8[14],xmm4[15],xmm8[15]
; SSE2-NEXT:    movdqa %xmm0, %xmm6
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm6 = xmm6[0],xmm8[0],xmm6[1],xmm8[1],xmm6[2],xmm8[2],xmm6[3],xmm8[3],xmm6[4],xmm8[4],xmm6[5],xmm8[5],xmm6[6],xmm8[6],xmm6[7],xmm8[7]
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm6 = xmm6[4],xmm4[4],xmm6[5],xmm4[5],xmm6[6],xmm4[6],xmm6[7],xmm4[7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm4 = xmm6[0,1,2,1]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm6 = xmm4[0,1,2,3,4,5,7,7]
; SSE2-NEXT:    movdqa {{.*#+}} xmm4 = [65535,65535,0,65535,65535,65535,0,65535]
; SSE2-NEXT:    movdqa %xmm4, %xmm7
; SSE2-NEXT:    pandn %xmm6, %xmm7
; SSE2-NEXT:    movdqa %xmm2, %xmm6
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm6 = xmm6[8],xmm8[8],xmm6[9],xmm8[9],xmm6[10],xmm8[10],xmm6[11],xmm8[11],xmm6[12],xmm8[12],xmm6[13],xmm8[13],xmm6[14],xmm8[14],xmm6[15],xmm8[15]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm6 = xmm6[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm6 = xmm6[0,3,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm6 = xmm6[1,0,2,3,4,5,6,7]
; SSE2-NEXT:    movdqa %xmm2, %xmm5
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm5 = xmm5[0],xmm8[0],xmm5[1],xmm8[1],xmm5[2],xmm8[2],xmm5[3],xmm8[3],xmm5[4],xmm8[4],xmm5[5],xmm8[5],xmm5[6],xmm8[6],xmm5[7],xmm8[7]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm5 = xmm5[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[0,3,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm5 = xmm5[1,0,2,3,4,5,6,7]
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm5 = xmm5[0],xmm6[0]
; SSE2-NEXT:    pand %xmm4, %xmm5
; SSE2-NEXT:    por %xmm7, %xmm5
; SSE2-NEXT:    packuswb %xmm0, %xmm5
; SSE2-NEXT:    movdqa {{.*#+}} xmm9 = [255,255,255,255,255,255,255,255]
; SSE2-NEXT:    pand %xmm9, %xmm2
; SSE2-NEXT:    movdqa {{.*#+}} xmm7 = [0,65535,65535,65535,0,65535,65535,65535]
; SSE2-NEXT:    movdqa %xmm7, %xmm6
; SSE2-NEXT:    pandn %xmm2, %xmm6
; SSE2-NEXT:    pand %xmm9, %xmm0
; SSE2-NEXT:    pand %xmm7, %xmm0
; SSE2-NEXT:    por %xmm6, %xmm0
; SSE2-NEXT:    packuswb %xmm0, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm5[0],xmm0[1],xmm5[1],xmm0[2],xmm5[2],xmm0[3],xmm5[3],xmm0[4],xmm5[4],xmm0[5],xmm5[5],xmm0[6],xmm5[6],xmm0[7],xmm5[7]
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm2 = xmm2[8],xmm8[8],xmm2[9],xmm8[9],xmm2[10],xmm8[10],xmm2[11],xmm8[11],xmm2[12],xmm8[12],xmm2[13],xmm8[13],xmm2[14],xmm8[14],xmm2[15],xmm8[15]
; SSE2-NEXT:    movdqa %xmm1, %xmm5
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm5 = xmm5[0],xmm8[0],xmm5[1],xmm8[1],xmm5[2],xmm8[2],xmm5[3],xmm8[3],xmm5[4],xmm8[4],xmm5[5],xmm8[5],xmm5[6],xmm8[6],xmm5[7],xmm8[7]
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm5 = xmm5[4],xmm2[4],xmm5[5],xmm2[5],xmm5[6],xmm2[6],xmm5[7],xmm2[7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm5[0,1,2,1]
; SSE2-NEXT:    pshufhw {{.*#+}} xmm2 = xmm2[0,1,2,3,4,5,7,7]
; SSE2-NEXT:    movdqa %xmm3, %xmm5
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm5 = xmm5[8],xmm8[8],xmm5[9],xmm8[9],xmm5[10],xmm8[10],xmm5[11],xmm8[11],xmm5[12],xmm8[12],xmm5[13],xmm8[13],xmm5[14],xmm8[14],xmm5[15],xmm8[15]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm5 = xmm5[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm5 = xmm5[0,3,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm5 = xmm5[1,0,2,3,4,5,6,7]
; SSE2-NEXT:    movdqa %xmm3, %xmm6
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm6 = xmm6[0],xmm8[0],xmm6[1],xmm8[1],xmm6[2],xmm8[2],xmm6[3],xmm8[3],xmm6[4],xmm8[4],xmm6[5],xmm8[5],xmm6[6],xmm8[6],xmm6[7],xmm8[7]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm6 = xmm6[3,1,2,3,4,5,6,7]
; SSE2-NEXT:    pshufd {{.*#+}} xmm6 = xmm6[0,3,2,3]
; SSE2-NEXT:    pshuflw {{.*#+}} xmm6 = xmm6[1,0,2,3,4,5,6,7]
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm6 = xmm6[0],xmm5[0]
; SSE2-NEXT:    pand %xmm4, %xmm6
; SSE2-NEXT:    pandn %xmm2, %xmm4
; SSE2-NEXT:    por %xmm6, %xmm4
; SSE2-NEXT:    packuswb %xmm0, %xmm4
; SSE2-NEXT:    pand %xmm9, %xmm3
; SSE2-NEXT:    pand %xmm9, %xmm1
; SSE2-NEXT:    pand %xmm7, %xmm1
; SSE2-NEXT:    pandn %xmm3, %xmm7
; SSE2-NEXT:    por %xmm7, %xmm1
; SSE2-NEXT:    packuswb %xmm0, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm4[0],xmm1[1],xmm4[1],xmm1[2],xmm4[2],xmm1[3],xmm4[3],xmm1[4],xmm4[4],xmm1[5],xmm4[5],xmm1[6],xmm4[6],xmm1[7],xmm4[7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: constant_pblendvb_avx2:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa {{.*#+}} xmm8 = <128,128,5,128,128,128,13,128,u,u,u,u,u,u,u,u>
; SSSE3-NEXT:    movdqa %xmm0, %xmm5
; SSSE3-NEXT:    pshufb %xmm8, %xmm5
; SSSE3-NEXT:    movdqa {{.*#+}} xmm6 = <1,3,128,7,9,11,128,15,u,u,u,u,u,u,u,u>
; SSSE3-NEXT:    movdqa %xmm2, %xmm7
; SSSE3-NEXT:    pshufb %xmm6, %xmm7
; SSSE3-NEXT:    por %xmm5, %xmm7
; SSSE3-NEXT:    movdqa {{.*#+}} xmm5 = <0,128,128,128,8,128,128,128,u,u,u,u,u,u,u,u>
; SSSE3-NEXT:    pshufb %xmm5, %xmm2
; SSSE3-NEXT:    movdqa {{.*#+}} xmm4 = <128,2,4,6,128,10,12,14,u,u,u,u,u,u,u,u>
; SSSE3-NEXT:    pshufb %xmm4, %xmm0
; SSSE3-NEXT:    por %xmm2, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm7[0],xmm0[1],xmm7[1],xmm0[2],xmm7[2],xmm0[3],xmm7[3],xmm0[4],xmm7[4],xmm0[5],xmm7[5],xmm0[6],xmm7[6],xmm0[7],xmm7[7]
; SSSE3-NEXT:    movdqa %xmm1, %xmm2
; SSSE3-NEXT:    pshufb %xmm8, %xmm2
; SSSE3-NEXT:    movdqa %xmm3, %xmm7
; SSSE3-NEXT:    pshufb %xmm6, %xmm7
; SSSE3-NEXT:    por %xmm2, %xmm7
; SSSE3-NEXT:    pshufb %xmm5, %xmm3
; SSSE3-NEXT:    pshufb %xmm4, %xmm1
; SSSE3-NEXT:    por %xmm3, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm7[0],xmm1[1],xmm7[1],xmm1[2],xmm7[2],xmm1[3],xmm7[3],xmm1[4],xmm7[4],xmm1[5],xmm7[5],xmm1[6],xmm7[6],xmm1[7],xmm7[7]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: constant_pblendvb_avx2:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm4
; SSE41-NEXT:    movaps {{.*#+}} xmm0 = [0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0]
; SSE41-NEXT:    pblendvb %xmm4, %xmm2
; SSE41-NEXT:    pblendvb %xmm1, %xmm3
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    movdqa %xmm3, %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: constant_pblendvb_avx2:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm8 = <128,128,5,128,128,128,13,128,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm8, %xmm2, %xmm4
; AVX1-NEXT:    vextractf128 $1, %ymm1, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm6 = <1,3,128,7,9,11,128,15,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm6, %xmm5, %xmm7
; AVX1-NEXT:    vpor %xmm4, %xmm7, %xmm4
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm7 = <0,128,128,128,8,128,128,128,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm7, %xmm5, %xmm5
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm3 = <128,2,4,6,128,10,12,14,u,u,u,u,u,u,u,u>
; AVX1-NEXT:    vpshufb %xmm3, %xmm2, %xmm2
; AVX1-NEXT:    vpor %xmm5, %xmm2, %xmm2
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3],xmm2[4],xmm4[4],xmm2[5],xmm4[5],xmm2[6],xmm4[6],xmm2[7],xmm4[7]
; AVX1-NEXT:    vpshufb %xmm8, %xmm0, %xmm4
; AVX1-NEXT:    vpshufb %xmm6, %xmm1, %xmm5
; AVX1-NEXT:    vpor %xmm4, %xmm5, %xmm4
; AVX1-NEXT:    vpshufb %xmm7, %xmm1, %xmm1
; AVX1-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; AVX1-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; AVX1-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: constant_pblendvb_avx2:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vmovdqa {{.*#+}} ymm2 = [0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0]
; AVX2-NEXT:    vpblendvb %ymm2, %ymm0, %ymm1, %ymm0
; AVX2-NEXT:    retq
entry:
  %select = select <32 x i1> <i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false>, <32 x i8> %xyzw, <32 x i8> %abcd
  ret <32 x i8> %select
}

declare <8 x float> @llvm.x86.avx.blendv.ps.256(<8 x float>, <8 x float>, <8 x float>)
declare <4 x double> @llvm.x86.avx.blendv.pd.256(<4 x double>, <4 x double>, <4 x double>)

;; 4 tests for shufflevectors that optimize to blend + immediate
define <4 x float> @blend_shufflevector_4xfloat(<4 x float> %a, <4 x float> %b) {
; SSE2-LABEL: blend_shufflevector_4xfloat:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_4xfloat:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_4xfloat:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: blend_shufflevector_4xfloat:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX-NEXT:    retq
entry:
  %select = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x float> %select
}

define <8 x float> @blend_shufflevector_8xfloat(<8 x float> %a, <8 x float> %b) {
; SSE2-LABEL: blend_shufflevector_8xfloat:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm3[3,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[0,2]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_8xfloat:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movss {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm3[3,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[0,2]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_8xfloat:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm2[1,2,3]
; SSE41-NEXT:    blendps {{.*#+}} xmm1 = xmm3[0,1],xmm1[2],xmm3[3]
; SSE41-NEXT:    retq
;
; AVX-LABEL: blend_shufflevector_8xfloat:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0],ymm1[1,2,3,4,5],ymm0[6],ymm1[7]
; AVX-NEXT:    retq
entry:
  %select = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 11, i32 12, i32 13, i32 6, i32 15>
  ret <8 x float> %select
}

define <4 x double> @blend_shufflevector_4xdouble(<4 x double> %a, <4 x double> %b) {
; SSE2-LABEL: blend_shufflevector_4xdouble:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm2 = xmm0[0],xmm2[1]
; SSE2-NEXT:    movapd %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_4xdouble:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm2 = xmm0[0],xmm2[1]
; SSSE3-NEXT:    movapd %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_4xdouble:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm2[1]
; SSE41-NEXT:    retq
;
; AVX-LABEL: blend_shufflevector_4xdouble:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0],ymm1[1],ymm0[2,3]
; AVX-NEXT:    retq
entry:
  %select = shufflevector <4 x double> %a, <4 x double> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x double> %select
}

define <4 x i64> @blend_shufflevector_4xi64(<4 x i64> %a, <4 x i64> %b) {
; SSE2-LABEL: blend_shufflevector_4xi64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsd {{.*#+}} xmm0 = xmm2[0],xmm0[1]
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_4xi64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd {{.*#+}} xmm0 = xmm2[0],xmm0[1]
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_4xi64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm0 = xmm2[0,1,2,3],xmm0[4,5,6,7]
; SSE41-NEXT:    movaps %xmm3, %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: blend_shufflevector_4xi64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vblendpd {{.*#+}} ymm0 = ymm1[0],ymm0[1],ymm1[2,3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: blend_shufflevector_4xi64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpblendd {{.*#+}} ymm0 = ymm1[0,1],ymm0[2,3],ymm1[4,5,6,7]
; AVX2-NEXT:    retq
entry:
  %select = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x i64> %select
}
