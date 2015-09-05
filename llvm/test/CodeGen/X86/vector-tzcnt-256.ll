; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

define <4 x i64> @testv4i64(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    movl $64, %ecx
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    vmovq %rax, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    cmoveq %rcx, %rax
; AVX1-NEXT:    vmovq %rax, %xmm0
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    movl $64, %ecx
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm1, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    vmovq %rax, %xmm1
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    cmoveq %rcx, %rax
; AVX2-NEXT:    vmovq %rax, %xmm0
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %in, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @testv4i64u(<4 x i64> %in) nounwind {
; AVX1-LABEL: testv4i64u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    vmovq %rax, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    vmovq %rax, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rax
; AVX1-NEXT:    bsfq %rax, %rax
; AVX1-NEXT:    vmovq %rax, %xmm0
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm1, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    vmovq %rax, %xmm1
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    vmovq %rax, %xmm2
; AVX2-NEXT:    vmovq %xmm0, %rax
; AVX2-NEXT:    bsfq %rax, %rax
; AVX2-NEXT:    vmovq %rax, %xmm0
; AVX2-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> %in, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %ecx
; AVX1-NEXT:    movl $32, %eax
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vmovd %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm1, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm1, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm0, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vmovd %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm0, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm0, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    cmovel %eax, %ecx
; AVX1-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %ecx
; AVX2-NEXT:    movl $32, %eax
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vmovd %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm1, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm1, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm0, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vmovd %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm0, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vpinsrd $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm0, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    cmovel %eax, %ecx
; AVX2-NEXT:    vpinsrd $3, %ecx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %in, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @testv8i32u(<8 x i32> %in) nounwind {
; AVX1-LABEL: testv8i32u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv8i32u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vmovd %xmm1, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrd $1, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vmovd %xmm0, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $2, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrd $3, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> %in, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %cx
; AVX1-NEXT:    movw $16, %ax
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vmovd %xmm1, %edx
; AVX1-NEXT:    bsfw %dx, %dx
; AVX1-NEXT:    cmovew %ax, %dx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vmovd %xmm0, %edx
; AVX1-NEXT:    bsfw %dx, %dx
; AVX1-NEXT:    cmovew %ax, %dx
; AVX1-NEXT:    vmovd %edx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    cmovew %ax, %cx
; AVX1-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %cx
; AVX2-NEXT:    movw $16, %ax
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vmovd %xmm1, %edx
; AVX2-NEXT:    bsfw %dx, %dx
; AVX2-NEXT:    cmovew %ax, %dx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vmovd %xmm0, %edx
; AVX2-NEXT:    bsfw %dx, %dx
; AVX2-NEXT:    cmovew %ax, %dx
; AVX2-NEXT:    vmovd %edx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $2, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $3, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $4, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $5, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $6, %ecx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    cmovew %ax, %cx
; AVX2-NEXT:    vpinsrw $7, %ecx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %in, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @testv16i16u(<16 x i16> %in) nounwind {
; AVX1-LABEL: testv16i16u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm1, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    bsfw %cx, %cx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm0, %eax
; AVX1-NEXT:    bsfw %ax, %ax
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vmovd %xmm1, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm1, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vmovd %xmm0, %ecx
; AVX2-NEXT:    bsfw %cx, %cx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm0, %eax
; AVX2-NEXT:    bsfw %ax, %ax
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> %in, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %edx
; AVX1-NEXT:    movl $32, %eax
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    movl $8, %ecx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpextrb $0, %xmm1, %esi
; AVX1-NEXT:    bsfl %esi, %esi
; AVX1-NEXT:    cmovel %eax, %esi
; AVX1-NEXT:    cmpl $32, %esi
; AVX1-NEXT:    cmovel %ecx, %esi
; AVX1-NEXT:    vmovd %esi, %xmm2
; AVX1-NEXT:    vpinsrb $1, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $2, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $3, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $4, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $5, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $6, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $7, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $8, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $9, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $10, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $11, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $12, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $13, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $14, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm1, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $15, %edx, %xmm2, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpextrb $0, %xmm0, %esi
; AVX1-NEXT:    bsfl %esi, %esi
; AVX1-NEXT:    cmovel %eax, %esi
; AVX1-NEXT:    cmpl $32, %esi
; AVX1-NEXT:    cmovel %ecx, %esi
; AVX1-NEXT:    vmovd %esi, %xmm2
; AVX1-NEXT:    vpinsrb $1, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $2, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $3, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $4, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $5, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $6, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $7, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $8, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $9, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $10, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $11, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $12, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $13, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $14, %edx, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm0, %edx
; AVX1-NEXT:    bsfl %edx, %edx
; AVX1-NEXT:    cmovel %eax, %edx
; AVX1-NEXT:    cmpl $32, %edx
; AVX1-NEXT:    cmovel %ecx, %edx
; AVX1-NEXT:    vpinsrb $15, %edx, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %edx
; AVX2-NEXT:    movl $32, %eax
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    movl $8, %ecx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpextrb $0, %xmm1, %esi
; AVX2-NEXT:    bsfl %esi, %esi
; AVX2-NEXT:    cmovel %eax, %esi
; AVX2-NEXT:    cmpl $32, %esi
; AVX2-NEXT:    cmovel %ecx, %esi
; AVX2-NEXT:    vmovd %esi, %xmm2
; AVX2-NEXT:    vpinsrb $1, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $2, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $3, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $4, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $5, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $6, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $7, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $8, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $9, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $10, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $11, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $12, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $13, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $14, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm1, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $15, %edx, %xmm2, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpextrb $0, %xmm0, %esi
; AVX2-NEXT:    bsfl %esi, %esi
; AVX2-NEXT:    cmovel %eax, %esi
; AVX2-NEXT:    cmpl $32, %esi
; AVX2-NEXT:    cmovel %ecx, %esi
; AVX2-NEXT:    vmovd %esi, %xmm2
; AVX2-NEXT:    vpinsrb $1, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $2, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $3, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $4, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $5, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $6, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $7, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $8, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $9, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $10, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $11, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $12, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $13, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $14, %edx, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm0, %edx
; AVX2-NEXT:    bsfl %edx, %edx
; AVX2-NEXT:    cmovel %eax, %edx
; AVX2-NEXT:    cmpl $32, %edx
; AVX2-NEXT:    cmovel %ecx, %edx
; AVX2-NEXT:    vpinsrb $15, %edx, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %in, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @testv32i8u(<32 x i8> %in) nounwind {
; AVX1-LABEL: testv32i8u:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm1, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX1-NEXT:    bsfl %ecx, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm0, %eax
; AVX1-NEXT:    bsfl %eax, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8u:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm1, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX2-NEXT:    bsfl %ecx, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm0, %eax
; AVX2-NEXT:    bsfl %eax, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> %in, i1 -1)
  ret <32 x i8> %out
}

define <4 x i64> @foldv4i64() nounwind {
; ALL-LABEL: foldv4i64:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,64,0]
; ALL-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 0)
  ret <4 x i64> %out
}

define <4 x i64> @foldv4i64u() nounwind {
; ALL-LABEL: foldv4i64u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,64,0]
; ALL-NEXT:    retq
  %out = call <4 x i64> @llvm.cttz.v4i64(<4 x i64> <i64 256, i64 -1, i64 0, i64 255>, i1 -1)
  ret <4 x i64> %out
}

define <8 x i32> @foldv8i32() nounwind {
; ALL-LABEL: foldv8i32:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,32,0,16,0,3,3]
; ALL-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 0)
  ret <8 x i32> %out
}

define <8 x i32> @foldv8i32u() nounwind {
; ALL-LABEL: foldv8i32u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,32,0,16,0,3,3]
; ALL-NEXT:    retq
  %out = call <8 x i32> @llvm.cttz.v8i32(<8 x i32> <i32 256, i32 -1, i32 0, i32 255, i32 -65536, i32 7, i32 24, i32 88>, i1 -1)
  ret <8 x i32> %out
}

define <16 x i16> @foldv16i16() nounwind {
; ALL-LABEL: foldv16i16:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,16,0,16,0,3,3,1,1,0,1,2,3,4,5]
; ALL-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 0)
  ret <16 x i16> %out
}

define <16 x i16> @foldv16i16u() nounwind {
; ALL-LABEL: foldv16i16u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,16,0,16,0,3,3,1,1,0,1,2,3,4,5]
; ALL-NEXT:    retq
  %out = call <16 x i16> @llvm.cttz.v16i16(<16 x i16> <i16 256, i16 -1, i16 0, i16 255, i16 -65536, i16 7, i16 24, i16 88, i16 -2, i16 254, i16 1, i16 2, i16 4, i16 8, i16 16, i16 32>, i1 -1)
  ret <16 x i16> %out
}

define <32 x i8> @foldv32i8() nounwind {
; ALL-LABEL: foldv32i8:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,0,0,0,0,0]
; ALL-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 0)
  ret <32 x i8> %out
}

define <32 x i8> @foldv32i8u() nounwind {
; ALL-LABEL: foldv32i8u:
; ALL:       # BB#0:
; ALL-NEXT:    vmovaps {{.*#+}} ymm0 = [8,0,8,0,8,0,3,3,1,1,0,1,2,3,4,5,6,7,8,8,7,6,5,4,3,2,1,0,0,0,0,0]
; ALL-NEXT:    retq
  %out = call <32 x i8> @llvm.cttz.v32i8(<32 x i8> <i8 256, i8 -1, i8 0, i8 255, i8 -65536, i8 7, i8 24, i8 88, i8 -2, i8 254, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 128, i8 256, i8 -256, i8 -128, i8 -64, i8 -32, i8 -16, i8 -8, i8 -4, i8 -2, i8 -1, i8 3, i8 5, i8 7, i8 127>, i1 -1)
  ret <32 x i8> %out
}

declare <4 x i64> @llvm.cttz.v4i64(<4 x i64>, i1)
declare <8 x i32> @llvm.cttz.v8i32(<8 x i32>, i1)
declare <16 x i16> @llvm.cttz.v16i16(<16 x i16>, i1)
declare <32 x i8> @llvm.cttz.v32i8(<32 x i8>, i1)
