; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl -mattr=+avx512cd | FileCheck %s --check-prefix=ALL --check-prefix=AVX512 --check-prefix=AVX512CD

define <8 x i64> @testv8i64(<8 x i64> %in) nounwind {
; ALL-LABEL: testv8i64:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vmovq %xmm1, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <8 x i64> @llvm.cttz.v8i64(<8 x i64> %in, i1 0)
  ret <8 x i64> %out
}

define <8 x i64> @testv8i64u(<8 x i64> %in) nounwind {
; ALL-LABEL: testv8i64u:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrq $1, %xmm1, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vmovq %xmm1, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm1
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrq $1, %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm2, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm2
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm2 = xmm2[0],xmm3[0]
; ALL-NEXT:    vpextrq $1, %xmm0, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm3
; ALL-NEXT:    vmovq %xmm0, %rax
; ALL-NEXT:    tzcntq %rax, %rax
; ALL-NEXT:    vmovq %rax, %xmm0
; ALL-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm3[0]
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <8 x i64> @llvm.cttz.v8i64(<8 x i64> %in, i1 -1)
  ret <8 x i64> %out
}

define <16 x i32> @testv16i32(<16 x i32> %in) nounwind {
; ALL-LABEL: testv16i32:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrd $1, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm2
; ALL-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; ALL-NEXT:    vpextrd $2, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; ALL-NEXT:    vpextrd $3, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm2
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <16 x i32> @llvm.cttz.v16i32(<16 x i32> %in, i1 0)
  ret <16 x i32> %out
}

define <16 x i32> @testv16i32u(<16 x i32> %in) nounwind {
; ALL-LABEL: testv16i32u:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti32x4 $3, %zmm0, %xmm1
; ALL-NEXT:    vpextrd $1, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm2
; ALL-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; ALL-NEXT:    vpextrd $2, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; ALL-NEXT:    vpextrd $3, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; ALL-NEXT:    vextracti32x4 $2, %zmm0, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm2
; ALL-NEXT:    vinserti128 $1, %xmm1, %ymm2, %ymm1
; ALL-NEXT:    vextracti32x4 $1, %zmm0, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrd $1, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vmovd %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrd $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $2, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrd $3, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrd $3, %eax, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vinserti64x4 $1, %ymm1, %zmm0, %zmm0
; ALL-NEXT:    retq
  %out = call <16 x i32> @llvm.cttz.v16i32(<16 x i32> %in, i1 -1)
  ret <16 x i32> %out
}

define <32 x i16> @testv32i16(<32 x i16> %in) nounwind {
; ALL-LABEL: testv32i16:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm0, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextracti128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm1, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
  %out = call <32 x i16> @llvm.cttz.v32i16(<32 x i16> %in, i1 0)
  ret <32 x i16> %out
}

define <32 x i16> @testv32i16u(<32 x i16> %in) nounwind {
; ALL-LABEL: testv32i16u:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm0, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm0, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextracti128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm2, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm2, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrw $1, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vmovd %xmm1, %ecx
; ALL-NEXT:    tzcntw %cx, %cx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrw $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $2, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $3, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $4, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $5, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $6, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrw $7, %xmm1, %eax
; ALL-NEXT:    tzcntw %ax, %ax
; ALL-NEXT:    vpinsrw $7, %eax, %xmm3, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
  %out = call <32 x i16> @llvm.cttz.v32i16(<32 x i16> %in, i1 -1)
  ret <32 x i16> %out
}

define <64 x i8> @testv64i8(<64 x i8> %in) nounwind {
; ALL-LABEL: testv64i8:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    movl $8, %eax
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpextrb $0, %xmm2, %edx
; ALL-NEXT:    tzcntl %edx, %edx
; ALL-NEXT:    cmpl $32, %edx
; ALL-NEXT:    cmovel %eax, %edx
; ALL-NEXT:    vmovd %edx, %xmm3
; ALL-NEXT:    vpinsrb $1, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $2, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $3, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $4, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $5, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $6, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $7, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $8, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $9, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $10, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $11, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $12, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $13, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $14, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $15, %ecx, %xmm3, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpextrb $0, %xmm0, %edx
; ALL-NEXT:    tzcntl %edx, %edx
; ALL-NEXT:    cmpl $32, %edx
; ALL-NEXT:    cmovel %eax, %edx
; ALL-NEXT:    vmovd %edx, %xmm3
; ALL-NEXT:    vpinsrb $1, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $2, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $3, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $4, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $5, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $6, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $7, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $8, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $9, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $10, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $11, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $12, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $13, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $14, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $15, %ecx, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextracti128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpextrb $0, %xmm2, %edx
; ALL-NEXT:    tzcntl %edx, %edx
; ALL-NEXT:    cmpl $32, %edx
; ALL-NEXT:    cmovel %eax, %edx
; ALL-NEXT:    vmovd %edx, %xmm3
; ALL-NEXT:    vpinsrb $1, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $2, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $3, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $4, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $5, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $6, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $7, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $8, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $9, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $10, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $11, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $12, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $13, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $14, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $15, %ecx, %xmm3, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpextrb $0, %xmm1, %edx
; ALL-NEXT:    tzcntl %edx, %edx
; ALL-NEXT:    cmpl $32, %edx
; ALL-NEXT:    cmovel %eax, %edx
; ALL-NEXT:    vmovd %edx, %xmm3
; ALL-NEXT:    vpinsrb $1, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $2, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $3, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $4, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $5, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $6, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $7, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $8, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $9, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $10, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $11, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $12, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $13, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $14, %ecx, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    cmpl $32, %ecx
; ALL-NEXT:    cmovel %eax, %ecx
; ALL-NEXT:    vpinsrb $15, %ecx, %xmm3, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
  %out = call <64 x i8> @llvm.cttz.v64i8(<64 x i8> %in, i1 0)
  ret <64 x i8> %out
}

define <64 x i8> @testv64i8u(<64 x i8> %in) nounwind {
; ALL-LABEL: testv64i8u:
; ALL:       ## BB#0:
; ALL-NEXT:    vextracti128 $1, %ymm0, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpextrb $0, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrb $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $7, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $8, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $9, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $10, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $11, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $12, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $13, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $14, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $15, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpextrb $0, %xmm0, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrb $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $7, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $8, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $9, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $10, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $11, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $12, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $13, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $14, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm0, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $15, %eax, %xmm3, %xmm0
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm0, %ymm0
; ALL-NEXT:    vextracti128 $1, %ymm1, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpextrb $0, %xmm2, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrb $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $7, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $8, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $9, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $10, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $11, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $12, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $13, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $14, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm2, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $15, %eax, %xmm3, %xmm2
; ALL-NEXT:    vpextrb $1, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpextrb $0, %xmm1, %ecx
; ALL-NEXT:    tzcntl %ecx, %ecx
; ALL-NEXT:    vmovd %ecx, %xmm3
; ALL-NEXT:    vpinsrb $1, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $2, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $2, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $3, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $3, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $4, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $4, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $5, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $5, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $6, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $6, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $7, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $7, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $8, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $8, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $9, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $9, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $10, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $10, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $11, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $11, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $12, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $12, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $13, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $13, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $14, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $14, %eax, %xmm3, %xmm3
; ALL-NEXT:    vpextrb $15, %xmm1, %eax
; ALL-NEXT:    tzcntl %eax, %eax
; ALL-NEXT:    vpinsrb $15, %eax, %xmm3, %xmm1
; ALL-NEXT:    vinserti128 $1, %xmm2, %ymm1, %ymm1
; ALL-NEXT:    retq
  %out = call <64 x i8> @llvm.cttz.v64i8(<64 x i8> %in, i1 -1)
  ret <64 x i8> %out
}

declare <8 x i64> @llvm.cttz.v8i64(<8 x i64>, i1)
declare <16 x i32> @llvm.cttz.v16i32(<16 x i32>, i1)
declare <32 x i16> @llvm.cttz.v32i16(<32 x i16>, i1)
declare <64 x i8> @llvm.cttz.v64i8(<64 x i8>, i1)
