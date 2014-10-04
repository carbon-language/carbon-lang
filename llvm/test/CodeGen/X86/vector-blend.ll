; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

; AVX128 tests:

define <4 x float> @vsel_float(<4 x float> %v1, <4 x float> %v2) {
; SSE2-LABEL: vsel_float:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_float:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
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
; SSE-LABEL: vsel_float2:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movss %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: vsel_float2:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovss %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %vsel = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x float> %v1, <4 x float> %v2
  ret <4 x float> %vsel
}

define <4 x i8> @vsel_4xi8(<4 x i8> %v1, <4 x i8> %v2) {
; SSE2-LABEL: vsel_4xi8:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_4xi8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
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
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_4xi16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
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
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
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
; SSE-LABEL: vsel_double:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movsd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: vsel_double:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovsd %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x double> %v1, <2 x double> %v2
  ret <2 x double> %vsel
}

define <2 x i64> @vsel_i64(<2 x i64> %v1, <2 x i64> %v2) {
; SSE-LABEL: vsel_i64:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movsd %xmm0, %xmm1
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: vsel_i64:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vmovsd %xmm0, %xmm1, %xmm0
; AVX-NEXT:    retq
entry:
  %vsel = select <2 x i1> <i1 true, i1 false>, <2 x i64> %v1, <2 x i64> %v2
  ret <2 x i64> %vsel
}

define <8 x i16> @vsel_8xi16(<8 x i16> %v1, <8 x i16> %v2) {
; SSE2-LABEL: vsel_8xi16:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_8xi16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_8xi16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm1 = xmm0[0],xmm1[1,2,3],xmm0[4],xmm1[5,6,7]
; SSE41-NEXT:    movdqa %xmm1, %xmm0
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
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm1
; SSE2-NEXT:    andps {{.*}}(%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm1
; SSSE3-NEXT:    andps {{.*}}(%rip), %xmm0
; SSSE3-NEXT:    orps %xmm1, %xmm0
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
; SSE-LABEL: vsel_float8:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movss %xmm0, %xmm2
; SSE-NEXT:    movss %xmm1, %xmm3
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    movaps %xmm3, %xmm1
; SSE-NEXT:    retq
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
; SSE-LABEL: vsel_i328:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movss %xmm0, %xmm2
; SSE-NEXT:    movss %xmm1, %xmm3
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    movaps %xmm3, %xmm1
; SSE-NEXT:    retq
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
; SSE2-NEXT:    movsd %xmm0, %xmm4
; SSE2-NEXT:    movsd %xmm2, %xmm6
; SSE2-NEXT:    movaps %xmm4, %xmm0
; SSE2-NEXT:    movaps %xmm5, %xmm1
; SSE2-NEXT:    movaps %xmm6, %xmm2
; SSE2-NEXT:    movaps %xmm7, %xmm3
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_double8:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd %xmm0, %xmm4
; SSSE3-NEXT:    movsd %xmm2, %xmm6
; SSSE3-NEXT:    movaps %xmm4, %xmm0
; SSSE3-NEXT:    movaps %xmm5, %xmm1
; SSSE3-NEXT:    movaps %xmm6, %xmm2
; SSSE3-NEXT:    movaps %xmm7, %xmm3
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_double8:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm4[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm1 = xmm5[0,1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm2 = xmm2[0],xmm6[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm3 = xmm7[0,1]
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
; SSE2-NEXT:    movsd %xmm0, %xmm4
; SSE2-NEXT:    movsd %xmm2, %xmm6
; SSE2-NEXT:    movaps %xmm4, %xmm0
; SSE2-NEXT:    movaps %xmm5, %xmm1
; SSE2-NEXT:    movaps %xmm6, %xmm2
; SSE2-NEXT:    movaps %xmm7, %xmm3
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: vsel_i648:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd %xmm0, %xmm4
; SSSE3-NEXT:    movsd %xmm2, %xmm6
; SSSE3-NEXT:    movaps %xmm4, %xmm0
; SSSE3-NEXT:    movaps %xmm5, %xmm1
; SSSE3-NEXT:    movaps %xmm6, %xmm2
; SSSE3-NEXT:    movaps %xmm7, %xmm3
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: vsel_i648:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendpd {{.*#+}} xmm0 = xmm0[0],xmm4[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm1 = xmm5[0,1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm2 = xmm2[0],xmm6[1]
; SSE41-NEXT:    blendpd {{.*#+}} xmm3 = xmm7[0,1]
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
; SSE-LABEL: vsel_double4:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movsd %xmm0, %xmm2
; SSE-NEXT:    movsd %xmm1, %xmm3
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    movaps %xmm3, %xmm1
; SSE-NEXT:    retq
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
; SSE-LABEL: constant_blendvpd_avx:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    movsd %xmm1, %xmm3
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    movaps %xmm3, %xmm1
; SSE-NEXT:    retq
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
; SSE2-NEXT:    movaps {{.*#+}} xmm4 = [4294967295,4294967295,4294967295,0]
; SSE2-NEXT:    andps %xmm4, %xmm2
; SSE2-NEXT:    movaps {{.*#+}} xmm5 = [0,0,0,4294967295]
; SSE2-NEXT:    andps %xmm5, %xmm0
; SSE2-NEXT:    orps %xmm2, %xmm0
; SSE2-NEXT:    andps %xmm4, %xmm3
; SSE2-NEXT:    andps %xmm5, %xmm1
; SSE2-NEXT:    orps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: constant_blendvps_avx:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movaps {{.*#+}} xmm4 = [4294967295,4294967295,4294967295,0]
; SSSE3-NEXT:    andps %xmm4, %xmm2
; SSSE3-NEXT:    movaps {{.*#+}} xmm5 = [0,0,0,4294967295]
; SSSE3-NEXT:    andps %xmm5, %xmm0
; SSSE3-NEXT:    orps %xmm2, %xmm0
; SSSE3-NEXT:    andps %xmm4, %xmm3
; SSSE3-NEXT:    andps %xmm5, %xmm1
; SSSE3-NEXT:    orps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: constant_blendvps_avx:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm2 = xmm2[0,1,2],xmm0[3]
; SSE41-NEXT:    blendps {{.*#+}} xmm3 = xmm3[0,1,2],xmm1[3]
; SSE41-NEXT:    movaps %xmm2, %xmm0
; SSE41-NEXT:    movaps %xmm3, %xmm1
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
; SSE2-NEXT:    movaps {{.*#+}} xmm4 = [255,255,0,255,0,0,0,255,255,255,0,255,0,0,0,255]
; SSE2-NEXT:    andps %xmm4, %xmm2
; SSE2-NEXT:    movaps {{.*#+}} xmm5 = [0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0]
; SSE2-NEXT:    andps %xmm5, %xmm0
; SSE2-NEXT:    orps %xmm2, %xmm0
; SSE2-NEXT:    andps %xmm4, %xmm3
; SSE2-NEXT:    andps %xmm5, %xmm1
; SSE2-NEXT:    orps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: constant_pblendvb_avx2:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movaps {{.*#+}} xmm4 = [255,255,0,255,0,0,0,255,255,255,0,255,0,0,0,255]
; SSSE3-NEXT:    andps %xmm4, %xmm2
; SSSE3-NEXT:    movaps {{.*#+}} xmm5 = [0,0,255,0,255,255,255,0,0,0,255,0,255,255,255,0]
; SSSE3-NEXT:    andps %xmm5, %xmm0
; SSSE3-NEXT:    orps %xmm2, %xmm0
; SSSE3-NEXT:    andps %xmm4, %xmm3
; SSSE3-NEXT:    andps %xmm5, %xmm1
; SSSE3-NEXT:    orps %xmm3, %xmm1
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
; AVX1-NEXT:    vandps {{.*}}(%rip), %ymm1, %ymm1
; AVX1-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    vorps %ymm1, %ymm0, %ymm0
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
; SSE2-NEXT:    movss %xmm0, %xmm2
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm3[3,0]
; SSE2-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[0,2]
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_8xfloat:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movss %xmm0, %xmm2
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[2,0],xmm3[3,0]
; SSSE3-NEXT:    shufps {{.*#+}} xmm3 = xmm3[0,1],xmm1[0,2]
; SSSE3-NEXT:    movaps %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_8xfloat:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    blendps {{.*#+}} xmm2 = xmm0[0],xmm2[1,2,3]
; SSE41-NEXT:    blendps {{.*#+}} xmm3 = xmm3[0,1],xmm1[2],xmm3[3]
; SSE41-NEXT:    movaps %xmm2, %xmm0
; SSE41-NEXT:    movaps %xmm3, %xmm1
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
; SSE2-NEXT:    movsd %xmm0, %xmm2
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_4xdouble:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd %xmm0, %xmm2
; SSSE3-NEXT:    movaps %xmm2, %xmm0
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
; SSE2-NEXT:    movsd %xmm2, %xmm0
; SSE2-NEXT:    movaps %xmm3, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: blend_shufflevector_4xi64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsd %xmm2, %xmm0
; SSSE3-NEXT:    movaps %xmm3, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: blend_shufflevector_4xi64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pblendw {{.*#+}} xmm2 = xmm2[0,1,2,3],xmm0[4,5,6,7]
; SSE41-NEXT:    movdqa %xmm2, %xmm0
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
