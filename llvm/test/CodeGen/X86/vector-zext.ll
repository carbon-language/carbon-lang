; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

define <8 x i32> @zext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: zext_8i16_to_8i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    # kill
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSE2-NEXT:    pand .LCPI0_0(%rip), %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: zext_8i16_to_8i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    pxor %xmm2, %xmm2
; SSSE3-NEXT:    # kill
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    pand .LCPI0_0(%rip), %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: zext_8i16_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pmovzxwd {{.*#+}} xmm0 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero
; SSE41-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSE41-NEXT:    pand .LCPI0_0(%rip), %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: zext_8i16_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: zext_8i16_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    retq
entry:
  %B = zext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @zext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: zext_4i32_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[0,1,1,3]
; SSE2-NEXT:    movdqa {{.*#+}} xmm3 = [4294967295,4294967295]
; SSE2-NEXT:    pand %xmm3, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,2,3,3]
; SSE2-NEXT:    pand %xmm3, %xmm1
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: zext_4i32_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[0,1,1,3]
; SSSE3-NEXT:    movdqa {{.*#+}} xmm3 = [4294967295,4294967295]
; SSSE3-NEXT:    pand %xmm3, %xmm2
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,2,3,3]
; SSSE3-NEXT:    pand %xmm3, %xmm1
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: zext_4i32_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovzxdq {{.*#+}} xmm2 = xmm0[0],zero,xmm0[1],zero
; SSE41-NEXT:    movdqa {{.*#+}} xmm3 = [4294967295,4294967295]
; SSE41-NEXT:    pand %xmm3, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,2,3,3]
; SSE41-NEXT:    pand %xmm3, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: zext_4i32_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhdq {{.*#+}} xmm1 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: zext_4i32_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxdq {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX2-NEXT:    retq
entry:
  %B = zext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

define <8 x i32> @zext_8i8_to_8i32(<8 x i8> %z) {
; SSE2-LABEL: zext_8i8_to_8i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3]
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [255,255,255,255]
; SSE2-NEXT:    pand %xmm1, %xmm2
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; SSE2-NEXT:    pand %xmm0, %xmm1
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: zext_8i8_to_8i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3]
; SSSE3-NEXT:    movdqa {{.*#+}} xmm1 = [255,255,255,255]
; SSSE3-NEXT:    pand %xmm1, %xmm2
; SSSE3-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    pand %xmm0, %xmm1
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: zext_8i8_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovzxwd {{.*#+}} xmm2 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; SSE41-NEXT:    movdqa {{.*#+}} xmm1 = [255,255,255,255]
; SSE41-NEXT:    pand %xmm1, %xmm2
; SSE41-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; SSE41-NEXT:    pand %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: zext_8i8_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: zext_8i8_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
entry:
  %t = zext <8 x i8> %z to <8 x i32>
  ret <8 x i32> %t
}

; PR17654
define <16 x i16> @zext_16i8_to_16i16(<16 x i8> %z) {
; SSE2-LABEL: zext_16i8_to_16i16:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    # kill
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSE2-NEXT:    pand .LCPI3_0(%rip), %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: zext_16i8_to_16i16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    pxor %xmm2, %xmm2
; SSSE3-NEXT:    # kill
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSSE3-NEXT:    pand .LCPI3_0(%rip), %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: zext_16i8_to_16i16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pmovzxbw %xmm1, %xmm0 {{.*#+}} xmm0 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero,xmm1[4],zero,xmm1[5],zero,xmm1[6],zero,xmm1[7],zero
; SSE41-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSE41-NEXT:    pand .LCPI3_0(%rip), %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: zext_16i8_to_16i16:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhbw {{.*#+}} xmm1 = xmm0[8],xmm1[8],xmm0[9],xmm1[9],xmm0[10],xmm1[10],xmm0[11],xmm1[11],xmm0[12],xmm1[12],xmm0[13],xmm1[13],xmm0[14],xmm1[14],xmm0[15],xmm1[15]
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: zext_16i8_to_16i16:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero,xmm0[8],zero,xmm0[9],zero,xmm0[10],zero,xmm0[11],zero,xmm0[12],zero,xmm0[13],zero,xmm0[14],zero,xmm0[15],zero
; AVX2-NEXT:    retq
entry:
  %t = zext <16 x i8> %z to <16 x i16>
  ret <16 x i16> %t
}

define <16 x i16> @load_zext_16i8_to_16i16(<16 x i8> *%ptr) {
; SSE2-LABEL: load_zext_16i8_to_16i16:
; SSE2:        # BB#0: # %entry
; SSE2-NEXT:   movdqa (%rdi), %xmm1
; SSE2-NEXT:   pxor %xmm2, %xmm2
; SSE2-NEXT:   movdqa %xmm1, %xmm0
; SSE2-NEXT:   punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:   punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSE2-NEXT:   pand .LCPI4_0(%rip), %xmm1
; SSE2-NEXT:   retq
;
; SSSE3-LABEL: load_zext_16i8_to_16i16:
; SSSE3:        # BB#0: # %entry
; SSSE3-NEXT:   movdqa (%rdi), %xmm1
; SSSE3-NEXT:   pxor %xmm2, %xmm2
; SSSE3-NEXT:   movdqa %xmm1, %xmm0
; SSSE3-NEXT:   punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:   punpckhbw {{.*#+}} xmm1 = xmm1[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; SSSE3-NEXT:   pand .LCPI4_0(%rip), %xmm1
; SSSE3-NEXT:   retq
;
; SSE41-LABEL: load_zext_16i8_to_16i16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovzxbw {{.*#+}} xmm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; SSE41-NEXT:    pmovzxbw {{.*#+}} xmm1 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: load_zext_16i8_to_16i16:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; AVX1-NEXT:    vpmovzxbw {{.*#+}} xmm1 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_zext_16i8_to_16i16:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxbw {{.*#+}} ymm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
; AVX2-NEXT:    retq
entry:
 %X = load <16 x i8>, <16 x i8>* %ptr
 %Y = zext <16 x i8> %X to <16 x i16>
 ret <16 x i16> %Y
}

define <8 x i32> @load_zext_8i16_to_8i32(<8 x i16> *%ptr) {
; SSE2-LABEL: load_zext_8i16_to_8i32:
; SSE2:          # BB#0: # %entry
; SSE2-NEXT:   movdqa (%rdi), %xmm1
; SSE2-NEXT:   pxor %xmm2, %xmm2
; SSE2-NEXT:   movdqa %xmm1, %xmm0
; SSE2-NEXT:   punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:   punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSE2-NEXT:   pand .LCPI5_0(%rip), %xmm1
; SSE2-NEXT:   retq
;
; SSSE3-LABEL: load_zext_8i16_to_8i32:
; SSSE3:        # BB#0: # %entry
; SSSE3-NEXT:   movdqa (%rdi), %xmm1
; SSSE3-NEXT:   pxor %xmm2, %xmm2
; SSSE3-NEXT:   movdqa %xmm1, %xmm0
; SSSE3-NEXT:   punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:   punpckhwd {{.*#+}} xmm1 = xmm1[4,4,5,5,6,6,7,7]
; SSSE3-NEXT:   pand .LCPI5_0(%rip), %xmm1
; SSSE3-NEXT:   retq
;
; SSE41-LABEL: load_zext_8i16_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovzxwd {{.*#+}} xmm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero
; SSE41-NEXT:    pmovzxwd {{.*#+}} xmm1 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: load_zext_8i16_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm1 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_zext_8i16_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero
; AVX2-NEXT:    retq
entry:
 %X = load <8 x i16>, <8 x i16>* %ptr
 %Y = zext <8 x i16> %X to <8 x i32>
 ret <8 x i32>%Y
}

define <4 x i64> @load_zext_4i32_to_4i64(<4 x i32> *%ptr) {
; SSE2-LABEL: load_zext_4i32_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa (%rdi), %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,1,1,3]
; SSE2-NEXT:    movdqa {{.*#+}} xmm2 = [4294967295,4294967295]
; SSE2-NEXT:    pand %xmm2, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,2,3,3]
; SSE2-NEXT:    pand %xmm2, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_zext_4i32_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa (%rdi), %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,1,1,3]
; SSSE3-NEXT:    movdqa {{.*#+}} xmm2 = [4294967295,4294967295]
; SSSE3-NEXT:    pand %xmm2, %xmm0
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,2,3,3]
; SSSE3-NEXT:    pand %xmm2, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_zext_4i32_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovzxdq {{.*#+}} xmm0 = mem[0],zero,mem[1],zero
; SSE41-NEXT:    pmovzxdq {{.*#+}} xmm1 = mem[0],zero,mem[1],zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: load_zext_4i32_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm0 = mem[0],zero,mem[1],zero
; AVX1-NEXT:    vpmovzxdq {{.*#+}} xmm1 = mem[0],zero,mem[1],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_zext_4i32_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovzxdq {{.*#+}} ymm0 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero
; AVX2-NEXT:    retq
entry:
 %X = load <4 x i32>, <4 x i32>* %ptr
 %Y = zext <4 x i32> %X to <4 x i64>
 ret <4 x i64>%Y
}

define <8 x i32> @shuf_zext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: shuf_zext_8i16_to_8i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    # kill
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: shuf_zext_8i16_to_8i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    pxor %xmm2, %xmm2
; SSSE3-NEXT:    # kill
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: shuf_zext_8i16_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pxor %xmm2, %xmm2
; SSE41-NEXT:    pmovzxwd {{.*#+}} xmm0 = xmm1[0],zero,xmm1[1],zero,xmm1[2],zero,xmm1[3],zero
; SSE41-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: shuf_zext_8i16_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX1-NEXT:    vpmovzxwd {{.*#+}} xmm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuf_zext_8i16_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    # kill
; AVX2-NEXT:    vpmovzxwd {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero,xmm0[4],zero,xmm0[5],zero,xmm0[6],zero,xmm0[7],zero
; AVX2-NEXT:    retq
entry:
  %B = shufflevector <8 x i16> %A, <8 x i16> zeroinitializer, <16 x i32> <i32 0, i32 8, i32 1, i32 8, i32 2, i32 8, i32 3, i32 8, i32 4, i32 8, i32 5, i32 8, i32 6, i32 8, i32 7, i32 8>
  %Z = bitcast <16 x i16> %B to <8 x i32>
  ret <8 x i32> %Z
}

define <4 x i64> @shuf_zext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: shuf_zext_4i32_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    pxor %xmm2, %xmm2
; SSE2-NEXT:    # kill
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE2-NEXT:    punpckhdq {{.*#+}} xmm1 = xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: shuf_zext_4i32_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    pxor %xmm2, %xmm2
; SSSE3-NEXT:    # kill
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSSE3-NEXT:    punpckhdq {{.*#+}} xmm1 = xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: shuf_zext_4i32_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pxor %xmm2, %xmm2
; SSE41-NEXT:    pmovzxdq {{.*#+}} xmm0 = xmm1[0],zero,xmm1[1],zero
; SSE41-NEXT:    punpckhdq {{.*#+}} xmm1 = xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: shuf_zext_4i32_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vinsertps {{.*#+}} xmm1 = xmm0[0],zero,xmm0[1],zero
; AVX1-NEXT:    vxorpd %xmm2, %xmm2, %xmm2
; AVX1-NEXT:    vblendpd {{.*#+}} xmm0 = xmm2[0],xmm0[1]
; AVX1-NEXT:    vpermilps {{.*#+}} xmm0 = xmm0[2,0,3,0]
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuf_zext_4i32_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    # kill
; AVX2-NEXT:    vpmovzxdq {{.*#+}} ymm0 = xmm0[0],zero,xmm0[1],zero,xmm0[2],zero,xmm0[3],zero
; AVX2-NEXT:    retq
entry:
  %B = shufflevector <4 x i32> %A, <4 x i32> zeroinitializer, <8 x i32> <i32 0, i32 4, i32 1, i32 4, i32 2, i32 4, i32 3, i32 4>
  %Z = bitcast <8 x i32> %B to <4 x i64>
  ret <4 x i64> %Z
}

define <8 x i32> @shuf_zext_8i8_to_8i32(<8 x i8> %A) {
; SSE2-LABEL: shuf_zext_8i8_to_8i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    pand .LCPI9_0(%rip), %xmm0
; SSE2-NEXT:    packuswb %xmm0, %xmm0
; SSE2-NEXT:    pxor %xmm1, %xmm1
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm0 = xmm0[4,4,5,5,6,6,7,7]
; SSE2-NEXT:    movdqa {{.*#+}} xmm1 = [0,255,255,255,0,255,255,255,0,255,255,255,0,255,255,255]
; SSE2-NEXT:    pandn %xmm0, %xmm1
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: shuf_zext_8i8_to_8i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; SSSE3-NEXT:    pxor %xmm2, %xmm2
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[4],zero,zero,zero,xmm1[5],zero,zero,zero,xmm1[6],zero,zero,zero,xmm1[7],zero,zero,zero
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: shuf_zext_8i8_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; SSE41-NEXT:    pmovzxbd {{.*#+}} xmm0 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; SSE41-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[4],zero,zero,zero,xmm1[5],zero,zero,zero,xmm1[6],zero,zero,zero,xmm1[7],zero,zero,zero
; SSE41-NEXT:    retq
;
; AVX1-LABEL: shuf_zext_8i8_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX1-NEXT:    vpmovzxbd {{.*#+}} xmm1 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero
; AVX1-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[4],zero,zero,zero,xmm0[5],zero,zero,zero,xmm0[6],zero,zero,zero,xmm0[7],zero,zero,zero
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: shuf_zext_8i8_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpshufb {{.*#+}} xmm0 = xmm0[0,2,4,6,8,10,12,14,u,u,u,u,u,u,u,u]
; AVX2-NEXT:    vpmovzxbd {{.*#+}} ymm0 = xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3],zero,zero,zero,xmm0[4],zero,zero,zero,xmm0[5],zero,zero,zero,xmm0[6],zero,zero,zero,xmm0[7],zero,zero,zero
; AVX2-NEXT:    retq
entry:
  %B = shufflevector <8 x i8> %A, <8 x i8> zeroinitializer, <32 x i32> <i32 0, i32 8, i32 8, i32 8, i32 1, i32 8, i32 8, i32 8, i32 2, i32 8, i32 8, i32 8, i32 3, i32 8, i32 8, i32 8, i32 4, i32 8, i32 8, i32 8, i32 5, i32 8, i32 8, i32 8, i32 6, i32 8, i32 8, i32 8, i32 7, i32 8, i32 8, i32 8>
  %Z = bitcast <32 x i8> %B to <8 x i32>
  ret <8 x i32> %Z
}
