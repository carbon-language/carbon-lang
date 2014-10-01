; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: sext_8i16_to_8i32:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:      ## kill: XMM0<def> XMM1<kill>
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    pslld $16, %xmm0
; SSE2-NEXT:    psrad $16, %xmm0
; SSE2-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    pslld $16, %xmm1
; SSE2-NEXT:    psrad $16, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_8i16_to_8i32:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:      ## kill: XMM0<def> XMM1<kill>
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    pslld $16, %xmm0
; SSSE3-NEXT:    psrad $16, %xmm0
; SSSE3-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    pslld $16, %xmm1
; SSSE3-NEXT:    psrad $16, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_8i16_to_8i32:
; SSE41:       ## BB#0:
; SSE41-NEXT:    movdqa %xmm0, %xmm1
; SSE41-NEXT:    pmovzxwd %xmm1, %xmm0
; SSE41-NEXT:    punpckhwd {{.*#+}} xmm1 = xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE41-NEXT:    pslld $16, %xmm1
; SSE41-NEXT:    psrad $16, %xmm1
; SSE41-NEXT:    pslld $16, %xmm0
; SSE41-NEXT:    psrad $16, %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: sext_4i32_to_4i64:
; SSE2:       ## BB#0:
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i32_to_4i64:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i32_to_4i64:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovzxdq %xmm0, %xmm1
; SSE41-NEXT:    pextrq $1, %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm3[0]
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_test1:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movq (%rdi), %xmm0
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    psrad $16, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test1:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movq (%rdi), %xmm0
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    psrad $16, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test1:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxwd (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE41-LABEL: load_sext_test2:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxbd (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_test3:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movsbq 1(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    movsbq (%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test3:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movsbq 1(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    movsbq (%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test3:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxbq (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_test4:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movswq 2(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    movswq (%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test4:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movswq 2(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    movswq (%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test4:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxwq (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_test5:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movslq 4(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    movslq (%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test5:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movslq 4(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    movslq (%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test5:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxdq (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_test6:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movq (%rdi), %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psraw $8, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test6:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movq (%rdi), %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psraw $8, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test6:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pmovsxbw (%rdi), %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: sext_4i1_to_4i64:
; SSE2:       ## BB#0:
; SSE2-NEXT:    pslld $31, %xmm0
; SSE2-NEXT:    psrad $31, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i1_to_4i64:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    pslld $31, %xmm0
; SSSE3-NEXT:    psrad $31, %xmm0
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i1_to_4i64:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pslld $31, %xmm0
; SSE41-NEXT:    psrad $31, %xmm0
; SSE41-NEXT:    pmovzxdq %xmm0, %xmm1
; SSE41-NEXT:    pextrq $1, %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm3[0]
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: sext_16i8_to_16i16:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movdqa (%rdi), %xmm1
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psllw $8, %xmm0
; SSE2-NEXT:    psraw $8, %xmm0
; SSE2-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
; SSE2-NEXT:    psllw $8, %xmm1
; SSE2-NEXT:    psraw $8, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_16i8_to_16i16:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movdqa (%rdi), %xmm1
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psllw $8, %xmm0
; SSSE3-NEXT:    psraw $8, %xmm0
; SSSE3-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
; SSSE3-NEXT:    psllw $8, %xmm1
; SSSE3-NEXT:    psraw $8, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_16i8_to_16i16:
; SSE41:       ## BB#0:
; SSE41-NEXT:    movdqa (%rdi), %xmm1
; SSE41-NEXT:    pmovzxbw %xmm1, %xmm0
; SSE41-NEXT:    punpckhbw {{.*#+}} xmm1 = xmm1[8],xmm0[8],xmm1[9],xmm0[9],xmm1[10],xmm0[10],xmm1[11],xmm0[11],xmm1[12],xmm0[12],xmm1[13],xmm0[13],xmm1[14],xmm0[14],xmm1[15],xmm0[15]
; SSE41-NEXT:    psllw $8, %xmm1
; SSE41-NEXT:    psraw $8, %xmm1
; SSE41-NEXT:    psllw $8, %xmm0
; SSE41-NEXT:    psraw $8, %xmm0
; SSE41-NEXT:    retq
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
; SSE2-LABEL: sext_4i8_to_4i64:
; SSE2:       ## BB#0:
; SSE2-NEXT:    pslld $24, %xmm0
; SSE2-NEXT:    psrad $24, %xmm0
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSE2-NEXT:    movd %xmm1, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    cltq
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i8_to_4i64:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    pslld $24, %xmm0
; SSSE3-NEXT:    psrad $24, %xmm0
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[0,0,1,0]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm1 = xmm1[1,1]
; SSSE3-NEXT:    movd %xmm1, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm1[0]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm0 = xmm0[1,1]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    cltq
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i8_to_4i64:
; SSE41:       ## BB#0:
; SSE41-NEXT:    pslld $24, %xmm0
; SSE41-NEXT:    psrad $24, %xmm0
; SSE41-NEXT:    pmovzxdq %xmm0, %xmm1
; SSE41-NEXT:    pextrq $1, %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm1, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,0,3,0]
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm3
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    cltq
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm3[0]
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
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
; SSE41-LABEL: load_sext_4i8_to_4i64:
; SSE41:       ## BB#0:
; SSE41-NEXT:    movd (%rdi), %xmm0
; SSE41-NEXT:    pmovzxbd %xmm0, %xmm1
; SSE41-NEXT:    pmovzxbq %xmm0, %xmm0
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    movsbq %al, %rax
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    movsbq %al, %rax
; SSE41-NEXT:    movd %rax, %xmm0
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,0,3,0]
; SSE41-NEXT:    pextrq $1, %xmm1, %rax
; SSE41-NEXT:    movsbq %al, %rax
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    movd %xmm1, %rax
; SSE41-NEXT:    movsbq %al, %rax
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE41-NEXT:    retq
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
; SSE2-LABEL: load_sext_4i16_to_4i64:
; SSE2:       ## BB#0:
; SSE2-NEXT:    movq (%rdi), %xmm1
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[0,0,1,0]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movswq %ax, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movswq %ax, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[2,0,3,0]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movswq %ax, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSE2-NEXT:    movd %xmm2, %rax
; SSE2-NEXT:    movswq %ax, %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_4i16_to_4i64:
; SSSE3:       ## BB#0:
; SSSE3-NEXT:    movq (%rdi), %xmm1
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[0,0,1,0]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movswq %ax, %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movswq %ax, %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm1[2,0,3,0]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movswq %ax, %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpckhqdq {{.*#+}} xmm2 = xmm2[1,1]
; SSSE3-NEXT:    movd %xmm2, %rax
; SSSE3-NEXT:    movswq %ax, %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_4i16_to_4i64:
; SSE41:       ## BB#0:
; SSE41-NEXT:    movq (%rdi), %xmm0
; SSE41-NEXT:    pmovzxwd %xmm0, %xmm1
; SSE41-NEXT:    pmovzxwq %xmm0, %xmm0
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    movswq %ax, %rax
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    movswq %ax, %rax
; SSE41-NEXT:    movd %rax, %xmm0
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; SSE41-NEXT:    pshufd {{.*#+}} xmm1 = xmm1[2,0,3,0]
; SSE41-NEXT:    pextrq $1, %xmm1, %rax
; SSE41-NEXT:    movswq %ax, %rax
; SSE41-NEXT:    movd %rax, %xmm2
; SSE41-NEXT:    movd %xmm1, %rax
; SSE41-NEXT:    movswq %ax, %rax
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE41-NEXT:    retq
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
