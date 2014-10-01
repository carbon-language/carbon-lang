; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=pentium4 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; SSE-LABEL: sext_8i16_to_8i32:
; SSE:       ## BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:      ## kill: XMM0<def> XMM1<kill>
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    pslld $16, %xmm0
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE-NEXT:    pslld $16, %xmm1
; SSE-NEXT:    psrad $16, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: sext_8i16_to_8i32:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_8i16_to_8i32:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    retq

  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @sext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; SSE-LABEL: sext_4i32_to_4i64:
; SSE:       ## BB#0:
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm2
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sext_4i32_to_4i64:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i32_to_4i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq

  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

define <4 x i32> @load_sext_test1(<4 x i16> *%ptr) {
; SSE-LABEL: load_sext_test1:
; SSE:       ## BB#0:
; SSE-NEXT:    movq (%rdi), %xmm0
; SSE-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE-NEXT:    psrad $16, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: load_sext_test1:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxwd (%rdi), %xmm0
; AVX-NEXT:    retq

 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i32>
 ret <4 x i32>%Y
}

define <4 x i32> @load_sext_test2(<4 x i8> *%ptr) {
; SSE2-LABEL: load_sext_test2:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movl (%rdi), %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shll $8, %ecx
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    pextrw $1, %xmm0, %edx
; SSE2-NEXT:    pinsrw $1, %ecx, %xmm0
; SSE2-NEXT:    pinsrw $3, %eax, %xmm0
; SSE2-NEXT:    movl %edx, %eax
; SSE2-NEXT:    shll $8, %eax
; SSE2-NEXT:    pinsrw $5, %eax, %xmm0
; SSE2-NEXT:    pinsrw $7, %edx, %xmm0
; SSE2-NEXT:    psrad $24, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test2:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movd (%rdi), %xmm0
; SSSE3-NEXT:    pshufb {{.*#+}} xmm0 = zero,zero,zero,xmm0[0],zero,zero,zero,xmm0[1],zero,zero,zero,xmm0[2],zero,zero,zero,xmm0[3]
; SSSE3-NEXT:    psrad $24, %xmm0
; SSSE3-NEXT:    retq
;
; AVX-LABEL: load_sext_test2:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxbd (%rdi), %xmm0
; AVX-NEXT:    retq
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i32>
 ret <4 x i32>%Y
}

define <2 x i64> @load_sext_test3(<2 x i8> *%ptr) {
; SSE-LABEL: load_sext_test3:
; SSE:       ## BB#0:
; SSE-NEXT:    movsbq 1(%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    movsbq (%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX-LABEL: load_sext_test3:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxbq (%rdi), %xmm0
; AVX-NEXT:    retq
 %X = load <2 x i8>* %ptr
 %Y = sext <2 x i8> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <2 x i64> @load_sext_test4(<2 x i16> *%ptr) {
; SSE-LABEL: load_sext_test4:
; SSE:       ## BB#0:
; SSE-NEXT:    movswq 2(%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    movswq (%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX-LABEL: load_sext_test4:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxwq (%rdi), %xmm0
; AVX-NEXT:    retq
 %X = load <2 x i16>* %ptr
 %Y = sext <2 x i16> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <2 x i64> @load_sext_test5(<2 x i32> *%ptr) {
; SSE-LABEL: load_sext_test5:
; SSE:       ## BB#0:
; SSE-NEXT:    movslq 4(%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    movslq (%rdi), %rax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE-NEXT:    retq
;
; AVX-LABEL: load_sext_test5:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxdq (%rdi), %xmm0
; AVX-NEXT:    retq
 %X = load <2 x i32>* %ptr
 %Y = sext <2 x i32> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <8 x i16> @load_sext_test6(<8 x i8> *%ptr) {
; SSE-LABEL: load_sext_test6:
; SSE:       ## BB#0:
; SSE-NEXT:    movq (%rdi), %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: load_sext_test6:
; AVX:       ## BB#0:
; AVX-NEXT:    vpmovsxbw (%rdi), %xmm0
; AVX-NEXT:    retq
 %X = load <8 x i8>* %ptr
 %Y = sext <8 x i8> %X to <8 x i16>
 ret <8 x i16>%Y
}

define <4 x i64> @sext_4i1_to_4i64(<4 x i1> %mask) {
; SSE-LABEL: sext_4i1_to_4i64:
; SSE:       ## BB#0:
; SSE-NEXT:    pslld $31, %xmm0
; SSE-NEXT:    psrad $31, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm2
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sext_4i1_to_4i64:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX1-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i1_to_4i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq
  %extmask = sext <4 x i1> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

define <16 x i16> @sext_16i8_to_16i16(<16 x i8> *%ptr) {
; SSE-LABEL: sext_16i8_to_16i16:
; SSE:       ## BB#0:
; SSE-NEXT:    movdqa (%rdi), %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE-NEXT:    psllw $8, %xmm0
; SSE-NEXT:    psraw $8, %xmm0
; SSE-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
; SSE-NEXT:    psllw $8, %xmm1
; SSE-NEXT:    psraw $8, %xmm1
; SSE-NEXT:    retq
;
; AVX1-LABEL: sext_16i8_to_16i16:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vmovdqa (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxbw %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_16i8_to_16i16:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vmovdqa (%rdi), %xmm0
; AVX2-NEXT:    vpmovsxbw %xmm0, %ymm0
; AVX2-NEXT:    retq
 %X = load <16 x i8>* %ptr
 %Y = sext <16 x i8> %X to <16 x i16>
 ret <16 x i16> %Y
}

define <4 x i64> @sext_4i8_to_4i64(<4 x i8> %mask) {
; SSE-LABEL: sext_4i8_to_4i64:
; SSE:       ## BB#0:
; SSE-NEXT:    pslld $24, %xmm0
; SSE-NEXT:    psrad $24, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm2
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE-NEXT:    movd %xmm1, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE-NEXT:    movd %xmm0, %rax
; SSE-NEXT:    cltq
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: sext_4i8_to_4i64:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpslld $24, %xmm0, %xmm0
; AVX1-NEXT:    vpsrad $24, %xmm0, %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i8_to_4i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpslld $24, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $24, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq
  %extmask = sext <4 x i8> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

define <4 x i64> @load_sext_4i8_to_4i64(<4 x i8> *%ptr) {
; SSE2-LABEL: load_sext_4i8_to_4i64:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movl (%rdi), %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pextrw $1, %xmm1, %ecx
; SSE2-NEXT:    pinsrw $0, %eax, %xmm1
; SSE2-NEXT:    movzbl %ah, %eax
; SSE2-NEXT:    pinsrw $2, %eax, %xmm1
; SSE2-NEXT:    pinsrw $4, %ecx, %xmm1
; SSE2-NEXT:    shrl $8, %ecx
; SSE2-NEXT:    pinsrw $6, %ecx, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[0,0,1,0]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movsbq %al, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movsbq %al, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[2,0,3,0]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movsbq %al, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movsbq %al, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_4i8_to_4i64:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movd (%rdi), %xmm1
; SSSE3-NEXT:    pshufb {{.*#+}} xmm1 = xmm1[0],zero,zero,zero,xmm1[1],zero,zero,zero,xmm1[2],zero,zero,zero,xmm1[3],zero,zero,zero
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[0,0,1,0]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movsbq %al, %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movsbq %al, %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[2,0,3,0]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movsbq %al, %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movsbq %al, %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSSE3-NEXT:    retq
;
; AVX1-LABEL: load_sext_4i8_to_4i64:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpmovsxbd (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_sext_4i8_to_4i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovsxbq (%rdi), %ymm0
; AVX2-NEXT:    retq
 %X = load <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i64>
 ret <4 x i64>%Y
}

define <4 x i64> @load_sext_4i16_to_4i64(<4 x i16> *%ptr) {
; SSE-LABEL: load_sext_4i16_to_4i64:
; SSE:       ## BB#0:
; SSE-NEXT:    movq (%rdi), %xmm1
; SSE-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[0,0,1,0]
; SSE-NEXT:    movd %xmm2, %rax
; SSE-NEXT:    movswq %ax, %rax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE-NEXT:    movd %xmm2, %rax
; SSE-NEXT:    movswq %ax, %rax
; SSE-NEXT:    movd %rax, %xmm2
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[2,0,3,0]
; SSE-NEXT:    movd %xmm2, %rax
; SSE-NEXT:    movswq %ax, %rax
; SSE-NEXT:    movd %rax, %xmm1
; SSE-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE-NEXT:    movd %xmm2, %rax
; SSE-NEXT:    movswq %ax, %rax
; SSE-NEXT:    movd %rax, %xmm2
; SSE-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE-NEXT:    retq
;
; AVX1-LABEL: load_sext_4i16_to_4i64:
; AVX1:       ## BB#0:
; AVX1-NEXT:    vpmovsxwd (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vmovhlps {{.*#+}} xmm0 = xmm0[1,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_sext_4i16_to_4i64:
; AVX2:       ## BB#0:
; AVX2-NEXT:    vpmovsxwq (%rdi), %ymm0
; AVX2-NEXT:    retq
 %X = load <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i64>
 ret <4 x i64>%Y
}
