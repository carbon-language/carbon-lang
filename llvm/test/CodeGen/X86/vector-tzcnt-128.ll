; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <2 x i64> @testv2i64(<2 x i64> %in) {
; SSE2-LABEL: testv2i64:
; SSE2:       # BB#0:
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    bsfq %rax, %rax
; SSE2-NEXT:    movl $64, %ecx
; SSE2-NEXT:    cmoveq %rcx, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    bsfq %rax, %rax
; SSE2-NEXT:    cmoveq %rcx, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv2i64:
; SSE3:       # BB#0:
; SSE3-NEXT:    movd %xmm0, %rax
; SSE3-NEXT:    bsfq %rax, %rax
; SSE3-NEXT:    movl $64, %ecx
; SSE3-NEXT:    cmoveq %rcx, %rax
; SSE3-NEXT:    movd %rax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE3-NEXT:    movd %xmm0, %rax
; SSE3-NEXT:    bsfq %rax, %rax
; SSE3-NEXT:    cmoveq %rcx, %rax
; SSE3-NEXT:    movd %rax, %xmm0
; SSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE3-NEXT:    movdqa %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv2i64:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    bsfq %rax, %rax
; SSSE3-NEXT:    movl $64, %ecx
; SSSE3-NEXT:    cmoveq %rcx, %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    bsfq %rax, %rax
; SSSE3-NEXT:    cmoveq %rcx, %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv2i64:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    bsfq %rax, %rax
; SSE41-NEXT:    movl $64, %ecx
; SSE41-NEXT:    cmoveq %rcx, %rax
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    bsfq %rax, %rax
; SSE41-NEXT:    cmoveq %rcx, %rax
; SSE41-NEXT:    movd %rax, %xmm0
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv2i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    bsfq %rax, %rax
; AVX-NEXT:    movl $64, %ecx
; AVX-NEXT:    cmoveq %rcx, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    bsfq %rax, %rax
; AVX-NEXT:    cmoveq %rcx, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %in, i1 0)
  ret <2 x i64> %out
}

define <2 x i64> @testv2i64u(<2 x i64> %in) {
; SSE2-LABEL: testv2i64u:
; SSE2:       # BB#0:
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    bsfq %rax, %rax
; SSE2-NEXT:    movd %rax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    movd %xmm0, %rax
; SSE2-NEXT:    bsfq %rax, %rax
; SSE2-NEXT:    movd %rax, %xmm0
; SSE2-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv2i64u:
; SSE3:       # BB#0:
; SSE3-NEXT:    movd %xmm0, %rax
; SSE3-NEXT:    bsfq %rax, %rax
; SSE3-NEXT:    movd %rax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE3-NEXT:    movd %xmm0, %rax
; SSE3-NEXT:    bsfq %rax, %rax
; SSE3-NEXT:    movd %rax, %xmm0
; SSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSE3-NEXT:    movdqa %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv2i64u:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    bsfq %rax, %rax
; SSSE3-NEXT:    movd %rax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSSE3-NEXT:    movd %xmm0, %rax
; SSSE3-NEXT:    bsfq %rax, %rax
; SSSE3-NEXT:    movd %rax, %xmm0
; SSSE3-NEXT:    punpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm0[0]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv2i64u:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrq $1, %xmm0, %rax
; SSE41-NEXT:    bsfq %rax, %rax
; SSE41-NEXT:    movd %rax, %xmm1
; SSE41-NEXT:    movd %xmm0, %rax
; SSE41-NEXT:    bsfq %rax, %rax
; SSE41-NEXT:    movd %rax, %xmm0
; SSE41-NEXT:    punpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv2i64u:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-NEXT:    bsfq %rax, %rax
; AVX-NEXT:    vmovq %rax, %xmm1
; AVX-NEXT:    vmovq %xmm0, %rax
; AVX-NEXT:    bsfq %rax, %rax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> %in, i1 -1)
  ret <2 x i64> %out
}

define <4 x i32> @testv4i32(<4 x i32> %in) {
; SSE2-LABEL: testv4i32:
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSE2-NEXT:    movd %xmm1, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movl $32, %ecx
; SSE2-NEXT:    cmovel %ecx, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSE2-NEXT:    movd %xmm2, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    cmovel %ecx, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    cmovel %ecx, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    cmovel %ecx, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv4i32:
; SSE3:       # BB#0:
; SSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSE3-NEXT:    movd %xmm1, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movl $32, %ecx
; SSE3-NEXT:    cmovel %ecx, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSE3-NEXT:    movd %xmm2, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    cmovel %ecx, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    cmovel %ecx, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    cmovel %ecx, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE3-NEXT:    movdqa %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv4i32:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSSE3-NEXT:    movd %xmm1, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movl $32, %ecx
; SSSE3-NEXT:    cmovel %ecx, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSSE3-NEXT:    movd %xmm2, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    cmovel %ecx, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    cmovel %ecx, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    cmovel %ecx, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv4i32:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrd $1, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    movl $32, %ecx
; SSE41-NEXT:    cmovel %ecx, %eax
; SSE41-NEXT:    movd %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    movd %edx, %xmm1
; SSE41-NEXT:    pinsrd $1, %eax, %xmm1
; SSE41-NEXT:    pextrd $2, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    cmovel %ecx, %eax
; SSE41-NEXT:    pinsrd $2, %eax, %xmm1
; SSE41-NEXT:    pextrd $3, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    cmovel %ecx, %eax
; SSE41-NEXT:    pinsrd $3, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrd $1, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    movl $32, %ecx
; AVX-NEXT:    cmovel %ecx, %eax
; AVX-NEXT:    vmovd %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vmovd %edx, %xmm1
; AVX-NEXT:    vpinsrd $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrd $2, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    cmovel %ecx, %eax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrd $3, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    cmovel %ecx, %eax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %in, i1 0)
  ret <4 x i32> %out
}

define <4 x i32> @testv4i32u(<4 x i32> %in) {
; SSE2-LABEL: testv4i32u:
; SSE2:       # BB#0:
; SSE2-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSE2-NEXT:    movd %xmm1, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSE2-NEXT:    movd %xmm2, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE2-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE2-NEXT:    movdqa %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv4i32u:
; SSE3:       # BB#0:
; SSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSE3-NEXT:    movd %xmm1, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSE3-NEXT:    movd %xmm2, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSE3-NEXT:    movdqa %xmm1, %xmm0
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv4i32u:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[3,1,2,3]
; SSSE3-NEXT:    movd %xmm1, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm2 = xmm0[1,1,2,3]
; SSSE3-NEXT:    movd %xmm2, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[2,3,0,1]
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
; SSSE3-NEXT:    punpckldq {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
; SSSE3-NEXT:    movdqa %xmm1, %xmm0
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv4i32u:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrd $1, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    movd %xmm0, %ecx
; SSE41-NEXT:    bsfl %ecx, %ecx
; SSE41-NEXT:    movd %ecx, %xmm1
; SSE41-NEXT:    pinsrd $1, %eax, %xmm1
; SSE41-NEXT:    pextrd $2, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrd $2, %eax, %xmm1
; SSE41-NEXT:    pextrd $3, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrd $3, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv4i32u:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrd $1, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vmovd %xmm0, %ecx
; AVX-NEXT:    bsfl %ecx, %ecx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrd $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrd $2, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrd $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrd $3, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrd $3, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %in, i1 -1)
  ret <4 x i32> %out
}

define <8 x i16> @testv8i16(<8 x i16> %in) {
; SSE2-LABEL: testv8i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    pextrw $7, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %cx
; SSE2-NEXT:    movw $16, %ax
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm1
; SSE2-NEXT:    pextrw $3, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE2-NEXT:    pextrw $5, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm3
; SSE2-NEXT:    pextrw $1, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm1
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE2-NEXT:    pextrw $6, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm2
; SSE2-NEXT:    pextrw $2, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm3
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE2-NEXT:    pextrw $4, %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm2
; SSE2-NEXT:    movd %xmm0, %ecx
; SSE2-NEXT:    bsfw %cx, %cx
; SSE2-NEXT:    cmovew %ax, %cx
; SSE2-NEXT:    movd %ecx, %xmm0
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv8i16:
; SSE3:       # BB#0:
; SSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %cx
; SSE3-NEXT:    movw $16, %ax
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm1
; SSE3-NEXT:    pextrw $3, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm2
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE3-NEXT:    pextrw $5, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm3
; SSE3-NEXT:    pextrw $1, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm1
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE3-NEXT:    pextrw $6, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm2
; SSE3-NEXT:    pextrw $2, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm3
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE3-NEXT:    pextrw $4, %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm2
; SSE3-NEXT:    movd %xmm0, %ecx
; SSE3-NEXT:    bsfw %cx, %cx
; SSE3-NEXT:    cmovew %ax, %cx
; SSE3-NEXT:    movd %ecx, %xmm0
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv8i16:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %cx
; SSSE3-NEXT:    movw $16, %ax
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm1
; SSSE3-NEXT:    pextrw $3, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSSE3-NEXT:    pextrw $5, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm3
; SSSE3-NEXT:    pextrw $1, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm1
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSSE3-NEXT:    pextrw $6, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm2
; SSSE3-NEXT:    pextrw $2, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm3
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSSE3-NEXT:    pextrw $4, %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm2
; SSSE3-NEXT:    movd %xmm0, %ecx
; SSSE3-NEXT:    bsfw %cx, %cx
; SSSE3-NEXT:    cmovew %ax, %cx
; SSSE3-NEXT:    movd %ecx, %xmm0
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv8i16:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrw $1, %xmm0, %eax
; SSE41-NEXT:    bsfw %ax, %cx
; SSE41-NEXT:    movw $16, %ax
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    movd %xmm0, %edx
; SSE41-NEXT:    bsfw %dx, %dx
; SSE41-NEXT:    cmovew %ax, %dx
; SSE41-NEXT:    movd %edx, %xmm1
; SSE41-NEXT:    pinsrw $1, %ecx, %xmm1
; SSE41-NEXT:    pextrw $2, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $2, %ecx, %xmm1
; SSE41-NEXT:    pextrw $3, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $3, %ecx, %xmm1
; SSE41-NEXT:    pextrw $4, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $4, %ecx, %xmm1
; SSE41-NEXT:    pextrw $5, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $5, %ecx, %xmm1
; SSE41-NEXT:    pextrw $6, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $6, %ecx, %xmm1
; SSE41-NEXT:    pextrw $7, %xmm0, %ecx
; SSE41-NEXT:    bsfw %cx, %cx
; SSE41-NEXT:    cmovew %ax, %cx
; SSE41-NEXT:    pinsrw $7, %ecx, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %cx
; AVX-NEXT:    movw $16, %ax
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vmovd %xmm0, %edx
; AVX-NEXT:    bsfw %dx, %dx
; AVX-NEXT:    cmovew %ax, %dx
; AVX-NEXT:    vmovd %edx, %xmm1
; AVX-NEXT:    vpinsrw $1, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $2, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $3, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $3, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $4, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $4, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $5, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $5, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $6, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $6, %ecx, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $7, %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    cmovew %ax, %cx
; AVX-NEXT:    vpinsrw $7, %ecx, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %in, i1 0)
  ret <8 x i16> %out
}

define <8 x i16> @testv8i16u(<8 x i16> %in) {
; SSE2-LABEL: testv8i16u:
; SSE2:       # BB#0:
; SSE2-NEXT:    pextrw $7, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pextrw $3, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE2-NEXT:    pextrw $5, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pextrw $1, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE2-NEXT:    pextrw $6, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pextrw $2, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE2-NEXT:    pextrw $4, %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    bsfw %ax, %ax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv8i16u:
; SSE3:       # BB#0:
; SSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pextrw $3, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE3-NEXT:    pextrw $5, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pextrw $1, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE3-NEXT:    pextrw $6, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pextrw $2, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE3-NEXT:    pextrw $4, %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    bsfw %ax, %ax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv8i16u:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pextrw $3, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSSE3-NEXT:    pextrw $5, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pextrw $1, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSSE3-NEXT:    pextrw $6, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pextrw $2, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSSE3-NEXT:    pextrw $4, %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    bsfw %ax, %ax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv8i16u:
; SSE41:       # BB#0:
; SSE41-NEXT:   pextrw $1, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   movd %xmm0, %ecx
; SSE41-NEXT:   bsfw %cx, %cx
; SSE41-NEXT:   movd %ecx, %xmm1
; SSE41-NEXT:   pinsrw $1, %eax, %xmm1
; SSE41-NEXT:   pextrw $2, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $2, %eax, %xmm1
; SSE41-NEXT:   pextrw $3, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $3, %eax, %xmm1
; SSE41-NEXT:   pextrw $4, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $4, %eax, %xmm1
; SSE41-NEXT:   pextrw $5, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $5, %eax, %xmm1
; SSE41-NEXT:   pextrw $6, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $6, %eax, %xmm1
; SSE41-NEXT:   pextrw $7, %xmm0, %eax
; SSE41-NEXT:   bsfw %ax, %ax
; SSE41-NEXT:   pinsrw $7, %eax, %xmm1
; SSE41-NEXT:   movdqa %xmm1, %xmm0
; SSE41-NEXT:   retq
;
; AVX-LABEL: testv8i16u:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vmovd %xmm0, %ecx
; AVX-NEXT:    bsfw %cx, %cx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $4, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $5, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $6, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-NEXT:    bsfw %ax, %ax
; AVX-NEXT:    vpinsrw $7, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> %in, i1 -1)
  ret <8 x i16> %out
}

define <16 x i8> @testv16i8(<16 x i8> %in) {
; SSE2-LABEL: testv16i8:
; SSE2:       # BB#0:
; SSE2:         pushq %rbp
; SSE2:         pushq %r14
; SSE2:         pushq %rbx
; SSE2:         movaps %xmm0, -16(%rsp)
; SSE2-NEXT:    movzbl -1(%rsp), %eax
; SSE2-NEXT:    bsfl %eax, %edx
; SSE2-NEXT:    movl $32, %eax
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    movl $8, %ecx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm0
; SSE2-NEXT:    movzbl -2(%rsp), %r14d
; SSE2-NEXT:    movzbl -3(%rsp), %ebx
; SSE2-NEXT:    movzbl -4(%rsp), %r9d
; SSE2-NEXT:    movzbl -5(%rsp), %edi
; SSE2-NEXT:    movzbl -6(%rsp), %r11d
; SSE2-NEXT:    movzbl -7(%rsp), %edx
; SSE2-NEXT:    movzbl -8(%rsp), %r8d
; SSE2-NEXT:    movzbl -9(%rsp), %esi
; SSE2-NEXT:    bsfl %esi, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    bsfl %edi, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm2
; SSE2-NEXT:    movzbl -10(%rsp), %edi
; SSE2-NEXT:    movzbl -11(%rsp), %esi
; SSE2-NEXT:    movzbl -12(%rsp), %r10d
; SSE2-NEXT:    movzbl -13(%rsp), %ebp
; SSE2-NEXT:    bsfl %ebp, %ebp
; SSE2-NEXT:    cmovel %eax, %ebp
; SSE2-NEXT:    cmpl $32, %ebp
; SSE2-NEXT:    cmovel %ecx, %ebp
; SSE2-NEXT:    movd %ebp, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    bsfl %ebx, %ebx
; SSE2-NEXT:    cmovel %eax, %ebx
; SSE2-NEXT:    cmpl $32, %ebx
; SSE2-NEXT:    cmovel %ecx, %ebx
; SSE2-NEXT:    movd %ebx, %xmm1
; SSE2-NEXT:    bsfl %esi, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:    bsfl %edx, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm3
; SSE2-NEXT:    movzbl -14(%rsp), %edx
; SSE2-NEXT:    movzbl -15(%rsp), %esi
; SSE2-NEXT:    bsfl %esi, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    bsfl %r14d, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    bsfl %edi, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm3
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:    bsfl %r11d, %esi
; SSE2-NEXT:    cmovel %eax, %esi
; SSE2-NEXT:    cmpl $32, %esi
; SSE2-NEXT:    cmovel %ecx, %esi
; SSE2-NEXT:    movd %esi, %xmm0
; SSE2-NEXT:    bsfl %edx, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE2-NEXT:    bsfl %r9d, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm0
; SSE2-NEXT:    bsfl %r10d, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm3
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:    bsfl %r8d, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm4
; SSE2-NEXT:    movzbl -16(%rsp), %edx
; SSE2-NEXT:    bsfl %edx, %edx
; SSE2-NEXT:    cmovel %eax, %edx
; SSE2-NEXT:    cmpl $32, %edx
; SSE2-NEXT:    cmovel %ecx, %edx
; SSE2-NEXT:    movd %edx, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    popq %rbx
; SSE2-NEXT:    popq %r14
; SSE2-NEXT:    popq %rbp
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv16i8:
; SSE3:       # BB#0:
; SSE3:         pushq %rbp
; SSE3:         pushq %r14
; SSE3:         pushq %rbx
; SSE3:         movaps %xmm0, -16(%rsp)
; SSE3-NEXT:    movzbl -1(%rsp), %eax
; SSE3-NEXT:    bsfl %eax, %edx
; SSE3-NEXT:    movl $32, %eax
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    movl $8, %ecx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm0
; SSE3-NEXT:    movzbl -2(%rsp), %r14d
; SSE3-NEXT:    movzbl -3(%rsp), %ebx
; SSE3-NEXT:    movzbl -4(%rsp), %r9d
; SSE3-NEXT:    movzbl -5(%rsp), %edi
; SSE3-NEXT:    movzbl -6(%rsp), %r11d
; SSE3-NEXT:    movzbl -7(%rsp), %edx
; SSE3-NEXT:    movzbl -8(%rsp), %r8d
; SSE3-NEXT:    movzbl -9(%rsp), %esi
; SSE3-NEXT:    bsfl %esi, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    bsfl %edi, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm2
; SSE3-NEXT:    movzbl -10(%rsp), %edi
; SSE3-NEXT:    movzbl -11(%rsp), %esi
; SSE3-NEXT:    movzbl -12(%rsp), %r10d
; SSE3-NEXT:    movzbl -13(%rsp), %ebp
; SSE3-NEXT:    bsfl %ebp, %ebp
; SSE3-NEXT:    cmovel %eax, %ebp
; SSE3-NEXT:    cmpl $32, %ebp
; SSE3-NEXT:    cmovel %ecx, %ebp
; SSE3-NEXT:    movd %ebp, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE3-NEXT:    bsfl %ebx, %ebx
; SSE3-NEXT:    cmovel %eax, %ebx
; SSE3-NEXT:    cmpl $32, %ebx
; SSE3-NEXT:    cmovel %ecx, %ebx
; SSE3-NEXT:    movd %ebx, %xmm1
; SSE3-NEXT:    bsfl %esi, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE3-NEXT:    bsfl %edx, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm3
; SSE3-NEXT:    movzbl -14(%rsp), %edx
; SSE3-NEXT:    movzbl -15(%rsp), %esi
; SSE3-NEXT:    bsfl %esi, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    bsfl %r14d, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm0
; SSE3-NEXT:    bsfl %edi, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm3
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE3-NEXT:    bsfl %r11d, %esi
; SSE3-NEXT:    cmovel %eax, %esi
; SSE3-NEXT:    cmpl $32, %esi
; SSE3-NEXT:    cmovel %ecx, %esi
; SSE3-NEXT:    movd %esi, %xmm0
; SSE3-NEXT:    bsfl %edx, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE3-NEXT:    bsfl %r9d, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm0
; SSE3-NEXT:    bsfl %r10d, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm3
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE3-NEXT:    bsfl %r8d, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm4
; SSE3-NEXT:    movzbl -16(%rsp), %edx
; SSE3-NEXT:    bsfl %edx, %edx
; SSE3-NEXT:    cmovel %eax, %edx
; SSE3-NEXT:    cmpl $32, %edx
; SSE3-NEXT:    cmovel %ecx, %edx
; SSE3-NEXT:    movd %edx, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE3-NEXT:    popq %rbx
; SSE3-NEXT:    popq %r14
; SSE3-NEXT:    popq %rbp
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv16i8:
; SSSE3:       # BB#0:
; SSSE3:         pushq %rbp
; SSSE3:         pushq %r14
; SSSE3:         pushq %rbx
; SSSE3:         movaps %xmm0, -16(%rsp)
; SSSE3-NEXT:    movzbl -1(%rsp), %eax
; SSSE3-NEXT:    bsfl %eax, %edx
; SSSE3-NEXT:    movl $32, %eax
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    movl $8, %ecx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm0
; SSSE3-NEXT:    movzbl -2(%rsp), %r14d
; SSSE3-NEXT:    movzbl -3(%rsp), %ebx
; SSSE3-NEXT:    movzbl -4(%rsp), %r9d
; SSSE3-NEXT:    movzbl -5(%rsp), %edi
; SSSE3-NEXT:    movzbl -6(%rsp), %r11d
; SSSE3-NEXT:    movzbl -7(%rsp), %edx
; SSSE3-NEXT:    movzbl -8(%rsp), %r8d
; SSSE3-NEXT:    movzbl -9(%rsp), %esi
; SSSE3-NEXT:    bsfl %esi, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    bsfl %edi, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm2
; SSSE3-NEXT:    movzbl -10(%rsp), %edi
; SSSE3-NEXT:    movzbl -11(%rsp), %esi
; SSSE3-NEXT:    movzbl -12(%rsp), %r10d
; SSSE3-NEXT:    movzbl -13(%rsp), %ebp
; SSSE3-NEXT:    bsfl %ebp, %ebp
; SSSE3-NEXT:    cmovel %eax, %ebp
; SSSE3-NEXT:    cmpl $32, %ebp
; SSSE3-NEXT:    cmovel %ecx, %ebp
; SSSE3-NEXT:    movd %ebp, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    bsfl %ebx, %ebx
; SSSE3-NEXT:    cmovel %eax, %ebx
; SSSE3-NEXT:    cmpl $32, %ebx
; SSSE3-NEXT:    cmovel %ecx, %ebx
; SSSE3-NEXT:    movd %ebx, %xmm1
; SSSE3-NEXT:    bsfl %esi, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSSE3-NEXT:    bsfl %edx, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm3
; SSSE3-NEXT:    movzbl -14(%rsp), %edx
; SSSE3-NEXT:    movzbl -15(%rsp), %esi
; SSSE3-NEXT:    bsfl %esi, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    bsfl %r14d, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    bsfl %edi, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm3
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSSE3-NEXT:    bsfl %r11d, %esi
; SSSE3-NEXT:    cmovel %eax, %esi
; SSSE3-NEXT:    cmpl $32, %esi
; SSSE3-NEXT:    cmovel %ecx, %esi
; SSSE3-NEXT:    movd %esi, %xmm0
; SSSE3-NEXT:    bsfl %edx, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSSE3-NEXT:    bsfl %r9d, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm0
; SSSE3-NEXT:    bsfl %r10d, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm3
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSSE3-NEXT:    bsfl %r8d, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm4
; SSSE3-NEXT:    movzbl -16(%rsp), %edx
; SSSE3-NEXT:    bsfl %edx, %edx
; SSSE3-NEXT:    cmovel %eax, %edx
; SSSE3-NEXT:    cmpl $32, %edx
; SSSE3-NEXT:    cmovel %ecx, %edx
; SSSE3-NEXT:    movd %edx, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    popq %rbx
; SSSE3-NEXT:    popq %r14
; SSSE3-NEXT:    popq %rbp
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv16i8:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrb $1, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %edx
; SSE41-NEXT:    movl $32, %eax
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    movl $8, %ecx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pextrb $0, %xmm0, %esi
; SSE41-NEXT:    bsfl %esi, %esi
; SSE41-NEXT:    cmovel %eax, %esi
; SSE41-NEXT:    cmpl $32, %esi
; SSE41-NEXT:    cmovel %ecx, %esi
; SSE41-NEXT:    movd %esi, %xmm1
; SSE41-NEXT:    pinsrb $1, %edx, %xmm1
; SSE41-NEXT:    pextrb $2, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $2, %edx, %xmm1
; SSE41-NEXT:    pextrb $3, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $3, %edx, %xmm1
; SSE41-NEXT:    pextrb $4, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $4, %edx, %xmm1
; SSE41-NEXT:    pextrb $5, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $5, %edx, %xmm1
; SSE41-NEXT:    pextrb $6, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $6, %edx, %xmm1
; SSE41-NEXT:    pextrb $7, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $7, %edx, %xmm1
; SSE41-NEXT:    pextrb $8, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $8, %edx, %xmm1
; SSE41-NEXT:    pextrb $9, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $9, %edx, %xmm1
; SSE41-NEXT:    pextrb $10, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $10, %edx, %xmm1
; SSE41-NEXT:    pextrb $11, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $11, %edx, %xmm1
; SSE41-NEXT:    pextrb $12, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $12, %edx, %xmm1
; SSE41-NEXT:    pextrb $13, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $13, %edx, %xmm1
; SSE41-NEXT:    pextrb $14, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $14, %edx, %xmm1
; SSE41-NEXT:    pextrb $15, %xmm0, %edx
; SSE41-NEXT:    bsfl %edx, %edx
; SSE41-NEXT:    cmovel %eax, %edx
; SSE41-NEXT:    cmpl $32, %edx
; SSE41-NEXT:    cmovel %ecx, %edx
; SSE41-NEXT:    pinsrb $15, %edx, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %edx
; AVX-NEXT:    movl $32, %eax
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    movl $8, %ecx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpextrb $0, %xmm0, %esi
; AVX-NEXT:    bsfl %esi, %esi
; AVX-NEXT:    cmovel %eax, %esi
; AVX-NEXT:    cmpl $32, %esi
; AVX-NEXT:    cmovel %ecx, %esi
; AVX-NEXT:    vmovd %esi, %xmm1
; AVX-NEXT:    vpinsrb $1, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $2, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $2, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $3, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $3, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $4, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $4, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $5, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $5, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $6, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $6, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $7, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $7, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $8, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $8, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $9, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $9, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $10, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $10, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $11, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $11, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $12, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $12, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $13, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $13, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $14, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $14, %edx, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $15, %xmm0, %edx
; AVX-NEXT:    bsfl %edx, %edx
; AVX-NEXT:    cmovel %eax, %edx
; AVX-NEXT:    cmpl $32, %edx
; AVX-NEXT:    cmovel %ecx, %edx
; AVX-NEXT:    vpinsrb $15, %edx, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %in, i1 0)
  ret <16 x i8> %out
}

define <16 x i8> @testv16i8u(<16 x i8> %in) {
; SSE2-LABEL: testv16i8u:
; SSE2:       # BB#0:
; SSE2:         pushq %rbx
; SSE2:         movaps %xmm0, -16(%rsp)
; SSE2-NEXT:    movzbl -1(%rsp), %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    movzbl -2(%rsp), %r11d
; SSE2-NEXT:    movzbl -3(%rsp), %eax
; SSE2-NEXT:    movzbl -4(%rsp), %r9d
; SSE2-NEXT:    movzbl -5(%rsp), %edi
; SSE2-NEXT:    movzbl -6(%rsp), %r10d
; SSE2-NEXT:    movzbl -7(%rsp), %ecx
; SSE2-NEXT:    movzbl -8(%rsp), %r8d
; SSE2-NEXT:    movzbl -9(%rsp), %edx
; SSE2-NEXT:    bsfl %edx, %edx
; SSE2-NEXT:    movd %edx, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    bsfl %edi, %edx
; SSE2-NEXT:    movd %edx, %xmm0
; SSE2-NEXT:    movzbl -10(%rsp), %edx
; SSE2-NEXT:    movzbl -11(%rsp), %esi
; SSE2-NEXT:    movzbl -12(%rsp), %edi
; SSE2-NEXT:    movzbl -13(%rsp), %ebx
; SSE2-NEXT:    bsfl %ebx, %ebx
; SSE2-NEXT:    movd %ebx, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    bsfl %esi, %eax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpcklbw %xmm0, %xmm3    # xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:    bsfl %ecx, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    movzbl -14(%rsp), %eax
; SSE2-NEXT:    movzbl -15(%rsp), %ecx
; SSE2-NEXT:    bsfl %ecx, %ecx
; SSE2-NEXT:    movd %ecx, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE2-NEXT:    bsfl %r11d, %ecx
; SSE2-NEXT:    movd %ecx, %xmm0
; SSE2-NEXT:    bsfl %edx, %ecx
; SSE2-NEXT:    movd %ecx, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:    bsfl %r10d, %ecx
; SSE2-NEXT:    movd %ecx, %xmm0
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; SSE2-NEXT:    bsfl %r9d, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    bsfl %edi, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:    bsfl %r8d, %eax
; SSE2-NEXT:    movd %eax, %xmm4
; SSE2-NEXT:    movzbl -16(%rsp), %eax
; SSE2-NEXT:    bsfl %eax, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    popq %rbx
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv16i8u:
; SSE3:       # BB#0:
; SSE3:         pushq %rbx
; SSE3:         movaps %xmm0, -16(%rsp)
; SSE3-NEXT:    movzbl -1(%rsp), %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    movzbl -2(%rsp), %r11d
; SSE3-NEXT:    movzbl -3(%rsp), %eax
; SSE3-NEXT:    movzbl -4(%rsp), %r9d
; SSE3-NEXT:    movzbl -5(%rsp), %edi
; SSE3-NEXT:    movzbl -6(%rsp), %r10d
; SSE3-NEXT:    movzbl -7(%rsp), %ecx
; SSE3-NEXT:    movzbl -8(%rsp), %r8d
; SSE3-NEXT:    movzbl -9(%rsp), %edx
; SSE3-NEXT:    bsfl %edx, %edx
; SSE3-NEXT:    movd %edx, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    bsfl %edi, %edx
; SSE3-NEXT:    movd %edx, %xmm0
; SSE3-NEXT:    movzbl -10(%rsp), %edx
; SSE3-NEXT:    movzbl -11(%rsp), %esi
; SSE3-NEXT:    movzbl -12(%rsp), %edi
; SSE3-NEXT:    movzbl -13(%rsp), %ebx
; SSE3-NEXT:    bsfl %ebx, %ebx
; SSE3-NEXT:    movd %ebx, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    bsfl %esi, %eax
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    punpcklbw %xmm0, %xmm3    # xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE3-NEXT:    bsfl %ecx, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    movzbl -14(%rsp), %eax
; SSE3-NEXT:    movzbl -15(%rsp), %ecx
; SSE3-NEXT:    bsfl %ecx, %ecx
; SSE3-NEXT:    movd %ecx, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE3-NEXT:    bsfl %r11d, %ecx
; SSE3-NEXT:    movd %ecx, %xmm0
; SSE3-NEXT:    bsfl %edx, %ecx
; SSE3-NEXT:    movd %ecx, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE3-NEXT:    bsfl %r10d, %ecx
; SSE3-NEXT:    movd %ecx, %xmm0
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; SSE3-NEXT:    bsfl %r9d, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    bsfl %edi, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE3-NEXT:    bsfl %r8d, %eax
; SSE3-NEXT:    movd %eax, %xmm4
; SSE3-NEXT:    movzbl -16(%rsp), %eax
; SSE3-NEXT:    bsfl %eax, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE3-NEXT:    popq %rbx
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv16i8u:
; SSSE3:       # BB#0:
; SSSE3:         pushq %rbx
; SSSE3:         movaps %xmm0, -16(%rsp)
; SSSE3-NEXT:    movzbl -1(%rsp), %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    movzbl -2(%rsp), %r11d
; SSSE3-NEXT:    movzbl -3(%rsp), %eax
; SSSE3-NEXT:    movzbl -4(%rsp), %r9d
; SSSE3-NEXT:    movzbl -5(%rsp), %edi
; SSSE3-NEXT:    movzbl -6(%rsp), %r10d
; SSSE3-NEXT:    movzbl -7(%rsp), %ecx
; SSSE3-NEXT:    movzbl -8(%rsp), %r8d
; SSSE3-NEXT:    movzbl -9(%rsp), %edx
; SSSE3-NEXT:    bsfl %edx, %edx
; SSSE3-NEXT:    movd %edx, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    bsfl %edi, %edx
; SSSE3-NEXT:    movd %edx, %xmm0
; SSSE3-NEXT:    movzbl -10(%rsp), %edx
; SSSE3-NEXT:    movzbl -11(%rsp), %esi
; SSSE3-NEXT:    movzbl -12(%rsp), %edi
; SSSE3-NEXT:    movzbl -13(%rsp), %ebx
; SSSE3-NEXT:    bsfl %ebx, %ebx
; SSSE3-NEXT:    movd %ebx, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    bsfl %esi, %eax
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    punpcklbw %xmm0, %xmm3    # xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSSE3-NEXT:    bsfl %ecx, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    movzbl -14(%rsp), %eax
; SSSE3-NEXT:    movzbl -15(%rsp), %ecx
; SSSE3-NEXT:    bsfl %ecx, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSSE3-NEXT:    bsfl %r11d, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm0
; SSSE3-NEXT:    bsfl %edx, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSSE3-NEXT:    bsfl %r10d, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm0
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; SSSE3-NEXT:    bsfl %r9d, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    bsfl %edi, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSSE3-NEXT:    bsfl %r8d, %eax
; SSSE3-NEXT:    movd %eax, %xmm4
; SSSE3-NEXT:    movzbl -16(%rsp), %eax
; SSSE3-NEXT:    bsfl %eax, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    popq %rbx
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv16i8u:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrb $1, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pextrb $0, %xmm0, %ecx
; SSE41-NEXT:    bsfl %ecx, %ecx
; SSE41-NEXT:    movd %ecx, %xmm1
; SSE41-NEXT:    pinsrb $1, %eax, %xmm1
; SSE41-NEXT:    pextrb $2, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $2, %eax, %xmm1
; SSE41-NEXT:    pextrb $3, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $3, %eax, %xmm1
; SSE41-NEXT:    pextrb $4, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $4, %eax, %xmm1
; SSE41-NEXT:    pextrb $5, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $5, %eax, %xmm1
; SSE41-NEXT:    pextrb $6, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $6, %eax, %xmm1
; SSE41-NEXT:    pextrb $7, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $7, %eax, %xmm1
; SSE41-NEXT:    pextrb $8, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $8, %eax, %xmm1
; SSE41-NEXT:    pextrb $9, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $9, %eax, %xmm1
; SSE41-NEXT:    pextrb $10, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $10, %eax, %xmm1
; SSE41-NEXT:    pextrb $11, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $11, %eax, %xmm1
; SSE41-NEXT:    pextrb $12, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $12, %eax, %xmm1
; SSE41-NEXT:    pextrb $13, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $13, %eax, %xmm1
; SSE41-NEXT:    pextrb $14, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $14, %eax, %xmm1
; SSE41-NEXT:    pextrb $15, %xmm0, %eax
; SSE41-NEXT:    bsfl %eax, %eax
; SSE41-NEXT:    pinsrb $15, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv16i8u:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-NEXT:    bsfl %ecx, %ecx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrb $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $4, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $5, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $6, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $7, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $8, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $9, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $10, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $11, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $12, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $13, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $14, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-NEXT:    bsfl %eax, %eax
; AVX-NEXT:    vpinsrb $15, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> %in, i1 -1)
  ret <16 x i8> %out
}

define <2 x i64> @foldv2i64() {
; SSE-LABEL: foldv2i64:
; SSE:       # BB#0:
; SSE-NEXT:    movl $8, %eax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv2i64:
; AVX:       # BB#0:
; AVX-NEXT:    movl $8, %eax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> <i64 256, i64 -1>, i1 0)
  ret <2 x i64> %out
}

define <2 x i64> @foldv2i64u() {
; SSE-LABEL: foldv2i64u:
; SSE:       # BB#0:
; SSE-NEXT:    movl $8, %eax
; SSE-NEXT:    movd %rax, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv2i64u:
; AVX:       # BB#0:
; AVX-NEXT:    movl $8, %eax
; AVX-NEXT:    vmovq %rax, %xmm0
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.cttz.v2i64(<2 x i64> <i64 256, i64 -1>, i1 -1)
  ret <2 x i64> %out
}

define <4 x i32> @foldv4i32() {
; SSE-LABEL: foldv4i32:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,32,0]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv4i32:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,32,0]
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> <i32 256, i32 -1, i32 0, i32 255>, i1 0)
  ret <4 x i32> %out
}

define <4 x i32> @foldv4i32u() {
; SSE-LABEL: foldv4i32u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,32,0]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv4i32u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,32,0]
; AVX-NEXT:    retq
  %out = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> <i32 256, i32 -1, i32 0, i32 255>, i1 -1)
  ret <4 x i32> %out
}

define <8 x i16> @foldv8i16() {
; SSE-LABEL: foldv8i16:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,16,0,16,0,3,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,16,0,16,0,3,3]
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88>, i1 0)
  ret <8 x i16> %out
}

define <8 x i16> @foldv8i16u() {
; SSE-LABEL: foldv8i16u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,16,0,16,0,3,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv8i16u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,16,0,16,0,3,3]
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.cttz.v8i16(<8 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88>, i1 -1)
  ret <8 x i16> %out
}

define <16 x i8> @foldv16i8() {
; SSE-LABEL: foldv16i8:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5]
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32>, i1 0)
  ret <16 x i8> %out
}

define <16 x i8> @foldv16i8u() {
; SSE-LABEL: foldv16i8u:
; SSE:       # BB#0:
; SSE-NEXT:    movaps {{.*#+}} xmm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5]
; SSE-NEXT:    retq
;
; AVX-LABEL: foldv16i8u:
; AVX:       # BB#0:
; AVX-NEXT:    vmovaps {{.*#+}} xmm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5]
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.cttz.v16i8(<16 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32>, i1 -1)
  ret <16 x i8> %out
}

declare <2 x i64> @llvm.cttz.v2i64(<2 x i64>, i1)
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1)
declare <8 x i16> @llvm.cttz.v8i16(<8 x i16>, i1)
declare <16 x i8> @llvm.cttz.v16i8(<16 x i8>, i1)
