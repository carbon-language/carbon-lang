; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -mattr=+popcnt | FileCheck --check-prefix=AVX --check-prefix=AVX-POPCNT %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -mattr=-popcnt | FileCheck --check-prefix=AVX --check-prefix=AVX-NOPOPCNT %s

define <4 x i32> @testv4i32(<4 x i32> %in) {
; AVX-POPCNT-LABEL: testv4i32:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vpextrd $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntl %ecx, %ecx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm1
; AVX-POPCNT-NEXT:    vpinsrd $1, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrd $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $2, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrd $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $3, %eax, %xmm1, %xmm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv4i32:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vpsrld $1, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpsubd %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vmovdqa {{.*#+}} xmm1 = [858993459,858993459,858993459,858993459]
; AVX-NOPOPCNT-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX-NOPOPCNT-NEXT:    vpsrld $2, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrld $4, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrld $8, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %in)
  ret <4 x i32> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) {
; AVX-POPCNT-LABEL: testv32i8:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $1, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $2, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $3, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $4, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $5, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $6, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $7, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $8, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $9, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $10, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $11, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $12, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $13, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $14, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $15, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX-POPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv32i8:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $1, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb %dl
; AVX-NOPOPCNT-NEXT:    andb $85, %dl
; AVX-NOPOPCNT-NEXT:    subb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $51, %dl
; AVX-NOPOPCNT-NEXT:    shrb $2, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    addb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb $4, %dl
; AVX-NOPOPCNT-NEXT:    addb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $15, %dl
; AVX-NOPOPCNT-NEXT:    movzbl %dl, %ecx
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $2, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $3, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $4, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $5, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $6, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $7, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $8, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $9, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $10, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $11, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $12, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $13, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $14, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $15, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb %dl
; AVX-NOPOPCNT-NEXT:    andb $85, %dl
; AVX-NOPOPCNT-NEXT:    subb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $51, %dl
; AVX-NOPOPCNT-NEXT:    shrb $2, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    addb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb $4, %dl
; AVX-NOPOPCNT-NEXT:    addb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $15, %dl
; AVX-NOPOPCNT-NEXT:    movzbl %dl, %ecx
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX-NOPOPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %in)
  ret <32 x i8> %out
}

define <4 x i64> @testv4i64(<4 x i64> %in) {
; AVX-POPCNT-LABEL: testv4i64:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-POPCNT-NEXT:    vpextrq $1, %xmm1, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm2
; AVX-POPCNT-NEXT:    vmovq %xmm1, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm1
; AVX-POPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX-POPCNT-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm2
; AVX-POPCNT-NEXT:    vmovq %xmm0, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm0
; AVX-POPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX-POPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv4i64:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrq $1, %xmm1, %rdx
; AVX-NOPOPCNT-NEXT:    movq %rdx, %rax
; AVX-NOPOPCNT-NEXT:    shrq %rax
; AVX-NOPOPCNT-NEXT:    movabsq $6148914691236517205, %r8 # imm = 0x5555555555555555
; AVX-NOPOPCNT-NEXT:    andq %r8, %rax
; AVX-NOPOPCNT-NEXT:    subq %rax, %rdx
; AVX-NOPOPCNT-NEXT:    movabsq $3689348814741910323, %rax # imm = 0x3333333333333333
; AVX-NOPOPCNT-NEXT:    movq %rdx, %rsi
; AVX-NOPOPCNT-NEXT:    andq %rax, %rsi
; AVX-NOPOPCNT-NEXT:    shrq $2, %rdx
; AVX-NOPOPCNT-NEXT:    andq %rax, %rdx
; AVX-NOPOPCNT-NEXT:    addq %rsi, %rdx
; AVX-NOPOPCNT-NEXT:    movq %rdx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $4, %rdi
; AVX-NOPOPCNT-NEXT:    addq %rdx, %rdi
; AVX-NOPOPCNT-NEXT:    movabsq $1085102592571150095, %rdx # imm = 0xF0F0F0F0F0F0F0F
; AVX-NOPOPCNT-NEXT:    andq %rdx, %rdi
; AVX-NOPOPCNT-NEXT:    movabsq $72340172838076673, %rsi # imm = 0x101010101010101
; AVX-NOPOPCNT-NEXT:    imulq %rsi, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $56, %rdi
; AVX-NOPOPCNT-NEXT:    vmovq %rdi, %xmm2
; AVX-NOPOPCNT-NEXT:    vmovq %xmm1, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq %rdi
; AVX-NOPOPCNT-NEXT:    andq %r8, %rdi
; AVX-NOPOPCNT-NEXT:    subq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    andq %rax, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $2, %rcx
; AVX-NOPOPCNT-NEXT:    andq %rax, %rcx
; AVX-NOPOPCNT-NEXT:    addq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $4, %rdi
; AVX-NOPOPCNT-NEXT:    addq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    andq %rdx, %rdi
; AVX-NOPOPCNT-NEXT:    imulq %rsi, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $56, %rdi
; AVX-NOPOPCNT-NEXT:    vmovq %rdi, %xmm1
; AVX-NOPOPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX-NOPOPCNT-NEXT:    vpextrq $1, %xmm0, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq %rdi
; AVX-NOPOPCNT-NEXT:    andq %r8, %rdi
; AVX-NOPOPCNT-NEXT:    subq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    andq %rax, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $2, %rcx
; AVX-NOPOPCNT-NEXT:    andq %rax, %rcx
; AVX-NOPOPCNT-NEXT:    addq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $4, %rdi
; AVX-NOPOPCNT-NEXT:    addq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    andq %rdx, %rdi
; AVX-NOPOPCNT-NEXT:    imulq %rsi, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $56, %rdi
; AVX-NOPOPCNT-NEXT:    vmovq %rdi, %xmm2
; AVX-NOPOPCNT-NEXT:    vmovq %xmm0, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    shrq %rdi
; AVX-NOPOPCNT-NEXT:    andq %r8, %rdi
; AVX-NOPOPCNT-NEXT:    subq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rdi
; AVX-NOPOPCNT-NEXT:    andq %rax, %rdi
; AVX-NOPOPCNT-NEXT:    shrq $2, %rcx
; AVX-NOPOPCNT-NEXT:    andq %rax, %rcx
; AVX-NOPOPCNT-NEXT:    addq %rdi, %rcx
; AVX-NOPOPCNT-NEXT:    movq %rcx, %rax
; AVX-NOPOPCNT-NEXT:    shrq $4, %rax
; AVX-NOPOPCNT-NEXT:    addq %rcx, %rax
; AVX-NOPOPCNT-NEXT:    andq %rdx, %rax
; AVX-NOPOPCNT-NEXT:    imulq %rsi, %rax
; AVX-NOPOPCNT-NEXT:    shrq $56, %rax
; AVX-NOPOPCNT-NEXT:    vmovq %rax, %xmm0
; AVX-NOPOPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX-NOPOPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %in)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) {
; AVX-POPCNT-LABEL: testv8i32:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-POPCNT-NEXT:    vpextrd $1, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX-POPCNT-NEXT:    popcntl %ecx, %ecx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrd $2, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrd $3, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX-POPCNT-NEXT:    vpextrd $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntl %ecx, %ecx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrd $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrd $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntl %eax, %eax
; AVX-POPCNT-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX-POPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv8i32:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrd $1, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl %edx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %edx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %edx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %edx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %edx, %ecx # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %ecx
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrd $2, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrd $3, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrd $1, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl %edx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %edx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %edx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %edx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %edx, %ecx # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %ecx
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrd $2, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrd $3, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX-NOPOPCNT-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX-NOPOPCNT-NEXT:    shrl $24, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX-NOPOPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %in)
  ret <8 x i32> %out
}

define <2 x i64> @testv2i64(<2 x i64> %in) {
; AVX-POPCNT-LABEL: testv2i64:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vpextrq $1, %xmm0, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm1
; AVX-POPCNT-NEXT:    vmovq %xmm0, %rax
; AVX-POPCNT-NEXT:    popcntq %rax, %rax
; AVX-POPCNT-NEXT:    vmovq %rax, %xmm0
; AVX-POPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv2i64:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vpsrlq $1, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpsubq %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vmovdqa {{.*#+}} xmm1 = [3689348814741910323,3689348814741910323]
; AVX-NOPOPCNT-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX-NOPOPCNT-NEXT:    vpsrlq $2, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrlq $4, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrlq $8, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrlq $16, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpsrlq $32, %xmm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %in)
  ret <2 x i64> %out
}

define <16 x i8> @testv16i8(<16 x i8> %in) {
; AVX-POPCNT-LABEL: testv16i8:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm1
; AVX-POPCNT-NEXT:    vpinsrb $1, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $2, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $4, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $5, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $6, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $7, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $8, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $9, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $10, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $11, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $12, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $13, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $14, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrb $15, %eax, %xmm1, %xmm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv16i8:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb %dl
; AVX-NOPOPCNT-NEXT:    andb $85, %dl
; AVX-NOPOPCNT-NEXT:    subb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $51, %dl
; AVX-NOPOPCNT-NEXT:    shrb $2, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    addb %dl, %cl
; AVX-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX-NOPOPCNT-NEXT:    shrb $4, %dl
; AVX-NOPOPCNT-NEXT:    addb %cl, %dl
; AVX-NOPOPCNT-NEXT:    andb $15, %dl
; AVX-NOPOPCNT-NEXT:    movzbl %dl, %ecx
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm1
; AVX-NOPOPCNT-NEXT:    vpinsrb $1, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $2, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $4, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $5, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $6, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $7, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $8, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $9, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $10, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $11, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $12, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $13, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $14, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb %cl
; AVX-NOPOPCNT-NEXT:    andb $85, %cl
; AVX-NOPOPCNT-NEXT:    subb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $51, %cl
; AVX-NOPOPCNT-NEXT:    shrb $2, %al
; AVX-NOPOPCNT-NEXT:    andb $51, %al
; AVX-NOPOPCNT-NEXT:    addb %cl, %al
; AVX-NOPOPCNT-NEXT:    movb %al, %cl
; AVX-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX-NOPOPCNT-NEXT:    addb %al, %cl
; AVX-NOPOPCNT-NEXT:    andb $15, %cl
; AVX-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX-NOPOPCNT-NEXT:    vpinsrb $15, %eax, %xmm1, %xmm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %in)
  ret <16 x i8> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) {
; AVX-POPCNT-LABEL: testv16i16:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $1, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $2, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $3, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $4, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $5, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $6, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $7, %xmm1, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-POPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-POPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX-POPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv16i16:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $1, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl %edx
; AVX-NOPOPCNT-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ch, %ecx # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $2, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $3, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $4, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $5, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $6, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $7, %xmm1, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl %edx
; AVX-NOPOPCNT-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ch, %ecx # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX-NOPOPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-NOPOPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX-NOPOPCNT-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %in)
  ret <16 x i16> %out
}

define <8 x i16> @testv8i16(<8 x i16> %in) {
; AVX-POPCNT-LABEL: testv8i16:
; AVX-POPCNT:       # BB#0:
; AVX-POPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-POPCNT-NEXT:    popcntw %cx, %cx
; AVX-POPCNT-NEXT:    vmovd %ecx, %xmm1
; AVX-POPCNT-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $2, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $4, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $5, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $6, %eax, %xmm1, %xmm1
; AVX-POPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-POPCNT-NEXT:    popcntw %ax, %ax
; AVX-POPCNT-NEXT:    vpinsrw $7, %eax, %xmm1, %xmm0
; AVX-POPCNT-NEXT:    retq
;
; AVX-NOPOPCNT-LABEL: testv8i16:
; AVX-NOPOPCNT:       # BB#0:
; AVX-NOPOPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    shrl %edx
; AVX-NOPOPCNT-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX-NOPOPCNT-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ch, %ecx # NOREX
; AVX-NOPOPCNT-NEXT:    vmovd %ecx, %xmm1
; AVX-NOPOPCNT-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $2, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $4, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $5, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $6, %eax, %xmm1, %xmm1
; AVX-NOPOPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    shrl %ecx
; AVX-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NOPOPCNT-NEXT:    vpinsrw $7, %eax, %xmm1, %xmm0
; AVX-NOPOPCNT-NEXT:    retq
  %out = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %in)
  ret <8 x i16> %out
}

declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
declare <32 x i8> @llvm.ctpop.v32i8(<32 x i8>)
declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>)
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>)
