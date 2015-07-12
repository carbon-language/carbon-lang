; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2
;
; Just one 32-bit run to make sure we do reasonable things there.
; RUN: llc < %s -mtriple=i686-unknown-unknown -mcpu=i686 -mattr=+sse4.1 | FileCheck %s --check-prefix=X32-SSE41

define <8 x i32> @sext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: sext_8i16_to_8i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3]
; SSE2-NEXT:    psrad $16, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSE2-NEXT:    psrad $16, %xmm1
; SSE2-NEXT:    movdqa %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_8i16_to_8i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3]
; SSSE3-NEXT:    psrad $16, %xmm2
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
; SSSE3-NEXT:    psrad $16, %xmm1
; SSSE3-NEXT:    movdqa %xmm2, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_8i16_to_8i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxwd %xmm0, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE41-NEXT:    pmovsxwd %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: sext_8i16_to_8i32:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxwd %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_8i16_to_8i32:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovsxwd %xmm0, %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: sext_8i16_to_8i32:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    pmovsxwd %xmm0, %xmm2
; X32-SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; X32-SSE41-NEXT:    pmovsxwd %xmm0, %xmm1
; X32-SSE41-NEXT:    movdqa %xmm2, %xmm0
; X32-SSE41-NEXT:    retl
entry:
  %B = sext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @sext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: sext_4i32_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i32_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movdqa %xmm0, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSSE3-NEXT:    movdqa %xmm1, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i32_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: sext_4i32_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i32_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: sext_4i32_to_4i64:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; X32-SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; X32-SSE41-NEXT:    movdqa %xmm2, %xmm0
; X32-SSE41-NEXT:    retl
entry:
  %B = sext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

define i32 @sext_2i8_to_i32(<16 x i8> %A) nounwind uwtable readnone ssp {
; SSE2-LABEL: sext_2i8_to_i32:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psraw $8, %xmm0
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_2i8_to_i32:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psraw $8, %xmm0
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_2i8_to_i32:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbw %xmm0, %xmm0
; SSE41-NEXT:    movd %xmm0, %eax
; SSE41-NEXT:    retq
;
; AVX-LABEL: sext_2i8_to_i32:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxbw %xmm0, %xmm0
; AVX-NEXT:    vmovd %xmm0, %eax
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: sext_2i8_to_i32:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41:         pmovsxbw %xmm0, %xmm0
; X32-SSE41-NEXT:    movd %xmm0, %eax
; X32-SSE41-NEXT:    popl %edx
; X32-SSE41-NEXT:    retl
entry:
  %Shuf = shufflevector <16 x i8> %A, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %Ex = sext <2 x i8> %Shuf to <2 x i16>
  %Bc = bitcast <2 x i16> %Ex to i32
  ret i32 %Bc
}

define <4 x i32> @load_sext_test1(<4 x i16> *%ptr) {
; SSE2-LABEL: load_sext_test1:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    psrad $16, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test1:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    psrad $16, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test1:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxwd (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test1:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxwd (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test1:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxwd (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <4 x i16>, <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i32>
 ret <4 x i32>%Y
}

define <4 x i32> @load_sext_test2(<4 x i8> *%ptr) {
; SSE2-LABEL: load_sext_test2:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    psrad $24, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test2:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    psrad $24, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test2:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbd (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test2:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxbd (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test2:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxbd (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <4 x i8>, <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i32>
 ret <4 x i32>%Y
}

define <2 x i64> @load_sext_test3(<2 x i8> *%ptr) {
; SSE2-LABEL: load_sext_test3:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movzwl (%rdi), %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrad $31, %xmm1
; SSE2-NEXT:    psrad $24, %xmm0
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test3:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movzwl (%rdi), %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    psrad $31, %xmm1
; SSSE3-NEXT:    psrad $24, %xmm0
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test3:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbq (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test3:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxbq (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test3:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxbq (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <2 x i8>, <2 x i8>* %ptr
 %Y = sext <2 x i8> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <2 x i64> @load_sext_test4(<2 x i16> *%ptr) {
; SSE2-LABEL: load_sext_test4:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrad $31, %xmm1
; SSE2-NEXT:    psrad $16, %xmm0
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test4:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movd {{.*#+}} xmm0 = mem[0],zero,zero,zero
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3]
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    psrad $31, %xmm1
; SSSE3-NEXT:    psrad $16, %xmm0
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test4:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxwq (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test4:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxwq (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test4:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxwq (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <2 x i16>, <2 x i16>* %ptr
 %Y = sext <2 x i16> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <2 x i64> @load_sext_test5(<2 x i32> *%ptr) {
; SSE2-LABEL: load_sext_test5:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    movdqa %xmm0, %xmm1
; SSE2-NEXT:    psrad $31, %xmm1
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test5:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSSE3-NEXT:    movdqa %xmm0, %xmm1
; SSSE3-NEXT:    psrad $31, %xmm1
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test5:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxdq (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test5:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxdq (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test5:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxdq (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <2 x i32>, <2 x i32>* %ptr
 %Y = sext <2 x i32> %X to <2 x i64>
 ret <2 x i64>%Y
}

define <8 x i16> @load_sext_test6(<8 x i8> *%ptr) {
; SSE2-LABEL: load_sext_test6:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psraw $8, %xmm0
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_test6:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psraw $8, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_test6:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbw (%rdi), %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: load_sext_test6:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpmovsxbw (%rdi), %xmm0
; AVX-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_test6:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxbw (%eax), %xmm0
; X32-SSE41-NEXT:    retl
entry:
 %X = load <8 x i8>, <8 x i8>* %ptr
 %Y = sext <8 x i8> %X to <8 x i16>
 ret <8 x i16>%Y
}

define <4 x i64> @sext_4i1_to_4i64(<4 x i1> %mask) {
; SSE2-LABEL: sext_4i1_to_4i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    pslld $31, %xmm0
; SSE2-NEXT:    psrad $31, %xmm0
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i1_to_4i64:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pslld $31, %xmm0
; SSSE3-NEXT:    psrad $31, %xmm0
; SSSE3-NEXT:    movdqa %xmm0, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSSE3-NEXT:    movdqa %xmm1, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i1_to_4i64:
; SSE41:       # BB#0:
; SSE41-NEXT:    pslld $31, %xmm0
; SSE41-NEXT:    psrad $31, %xmm0
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: sext_4i1_to_4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX1-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i1_to_4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpslld $31, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $31, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: sext_4i1_to_4i64:
; X32-SSE41:       # BB#0:
; X32-SSE41-NEXT:    pslld $31, %xmm0
; X32-SSE41-NEXT:    psrad $31, %xmm0
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; X32-SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; X32-SSE41-NEXT:    movdqa %xmm2, %xmm0
; X32-SSE41-NEXT:    retl
  %extmask = sext <4 x i1> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

define <16 x i16> @sext_16i8_to_16i16(<16 x i8> *%ptr) {
; SSE2-LABEL: sext_16i8_to_16i16:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psraw $8, %xmm0
; SSE2-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:    psraw $8, %xmm1
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_16i8_to_16i16:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movq {{.*#+}} xmm0 = mem[0],zero
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psraw $8, %xmm0
; SSSE3-NEXT:    movq {{.*#+}} xmm1 = mem[0],zero
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSSE3-NEXT:    psraw $8, %xmm1
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_16i8_to_16i16:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbw (%rdi), %xmm0
; SSE41-NEXT:    pmovsxbw 8(%rdi), %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: sext_16i8_to_16i16:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovsxbw (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxbw 8(%rdi), %xmm1
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_16i8_to_16i16:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovsxbw (%rdi), %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: sext_16i8_to_16i16:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxbw (%eax), %xmm0
; X32-SSE41-NEXT:    pmovsxbw 8(%eax), %xmm1
; X32-SSE41-NEXT:    retl
entry:
 %X = load <16 x i8>, <16 x i8>* %ptr
 %Y = sext <16 x i8> %X to <16 x i16>
 ret <16 x i16> %Y
}

define <4 x i64> @sext_4i8_to_4i64(<4 x i8> %mask) {
; SSE2-LABEL: sext_4i8_to_4i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    pslld $24, %xmm0
; SSE2-NEXT:    psrad $24, %xmm0
; SSE2-NEXT:    movdqa %xmm0, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm2
; SSE2-NEXT:    psrad $31, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: sext_4i8_to_4i64:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pslld $24, %xmm0
; SSSE3-NEXT:    psrad $24, %xmm0
; SSSE3-NEXT:    movdqa %xmm0, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,0,1]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; SSSE3-NEXT:    movdqa %xmm1, %xmm2
; SSSE3-NEXT:    psrad $31, %xmm2
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: sext_4i8_to_4i64:
; SSE41:       # BB#0:
; SSE41-NEXT:    pslld $24, %xmm0
; SSE41-NEXT:    psrad $24, %xmm0
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; SSE41-NEXT:    movdqa %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: sext_4i8_to_4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpslld $24, %xmm0, %xmm0
; AVX1-NEXT:    vpsrad $24, %xmm0, %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: sext_4i8_to_4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpslld $24, %xmm0, %xmm0
; AVX2-NEXT:    vpsrad $24, %xmm0, %xmm0
; AVX2-NEXT:    vpmovsxdq %xmm0, %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: sext_4i8_to_4i64:
; X32-SSE41:       # BB#0:
; X32-SSE41-NEXT:    pslld $24, %xmm0
; X32-SSE41-NEXT:    psrad $24, %xmm0
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm2
; X32-SSE41-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; X32-SSE41-NEXT:    pmovsxdq %xmm0, %xmm1
; X32-SSE41-NEXT:    movdqa %xmm2, %xmm0
; X32-SSE41-NEXT:    retl
  %extmask = sext <4 x i8> %mask to <4 x i64>
  ret <4 x i64> %extmask
}

define <4 x i64> @load_sext_4i8_to_4i64(<4 x i8> *%ptr) {
; SSE2-LABEL: load_sext_4i8_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movsbq 1(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    movsbq (%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    movsbq 3(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    movsbq 2(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_4i8_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movsbq 1(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    movsbq (%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSSE3-NEXT:    movsbq 3(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    movsbq 2(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_4i8_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxbq (%rdi), %xmm0
; SSE41-NEXT:    pmovsxbq 2(%rdi), %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: load_sext_4i8_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovsxbd (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_sext_4i8_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovsxbq (%rdi), %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_4i8_to_4i64:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxbq (%eax), %xmm0
; X32-SSE41-NEXT:    pmovsxbq 2(%eax), %xmm1
; X32-SSE41-NEXT:    retl
entry:
 %X = load <4 x i8>, <4 x i8>* %ptr
 %Y = sext <4 x i8> %X to <4 x i64>
 ret <4 x i64>%Y
}

define <4 x i64> @load_sext_4i16_to_4i64(<4 x i16> *%ptr) {
; SSE2-LABEL: load_sext_4i16_to_4i64:
; SSE2:       # BB#0: # %entry
; SSE2-NEXT:    movswq 2(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    movswq (%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE2-NEXT:    movswq 6(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm2
; SSE2-NEXT:    movswq 4(%rdi), %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: load_sext_4i16_to_4i64:
; SSSE3:       # BB#0: # %entry
; SSSE3-NEXT:    movswq 2(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    movswq (%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSSE3-NEXT:    movswq 6(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm2
; SSSE3-NEXT:    movswq 4(%rdi), %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: load_sext_4i16_to_4i64:
; SSE41:       # BB#0: # %entry
; SSE41-NEXT:    pmovsxwq (%rdi), %xmm0
; SSE41-NEXT:    pmovsxwq 4(%rdi), %xmm1
; SSE41-NEXT:    retq
;
; AVX1-LABEL: load_sext_4i16_to_4i64:
; AVX1:       # BB#0: # %entry
; AVX1-NEXT:    vpmovsxwd (%rdi), %xmm0
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm1
; AVX1-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; AVX1-NEXT:    vpmovsxdq %xmm0, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm0, %ymm1, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: load_sext_4i16_to_4i64:
; AVX2:       # BB#0: # %entry
; AVX2-NEXT:    vpmovsxwq (%rdi), %ymm0
; AVX2-NEXT:    retq
;
; X32-SSE41-LABEL: load_sext_4i16_to_4i64:
; X32-SSE41:       # BB#0: # %entry
; X32-SSE41-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-SSE41-NEXT:    pmovsxwq (%eax), %xmm0
; X32-SSE41-NEXT:    pmovsxwq 4(%eax), %xmm1
; X32-SSE41-NEXT:    retl
entry:
 %X = load <4 x i16>, <4 x i16>* %ptr
 %Y = sext <4 x i16> %X to <4 x i64>
 ret <4 x i64>%Y
}
