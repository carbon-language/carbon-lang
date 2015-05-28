; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 -mattr=+popcnt | FileCheck --check-prefix=AVX2 --check-prefix=AVX2-POPCNT %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 -mattr=-popcnt | FileCheck --check-prefix=AVX2 --check-prefix=AVX2-NOPOPCNT %s

; When avx2 is enabled, we should always generate the same code regardless
; of popcnt instruction availability.

define <32 x i8> @testv32i8(<32 x i8> %in) {
; AVX2-POPCNT-LABEL: testv32i8:
; AVX2-POPCNT:       # BB#0:
; AVX2-POPCNT-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-POPCNT-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX2-POPCNT-NEXT:    popcntw %cx, %cx
; AVX2-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-POPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $2, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $3, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $4, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $5, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $6, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $7, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $8, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $9, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $10, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $11, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $12, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $13, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $14, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $15, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX2-POPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX2-POPCNT-NEXT:    popcntw %cx, %cx
; AVX2-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-POPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX2-POPCNT-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-POPCNT-NEXT:    retq
;
; AVX2-NOPOPCNT-LABEL: testv32i8:
; AVX2-NOPOPCNT:       # BB#0:
; AVX2-NOPOPCNT-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NOPOPCNT-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    shrb %dl
; AVX2-NOPOPCNT-NEXT:    andb $85, %dl
; AVX2-NOPOPCNT-NEXT:    subb %dl, %cl
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    andb $51, %dl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    addb %dl, %cl
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %dl
; AVX2-NOPOPCNT-NEXT:    addb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    andb $15, %dl
; AVX2-NOPOPCNT-NEXT:    movzbl %dl, %ecx
; AVX2-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $2, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $3, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $4, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $5, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $6, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $7, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $8, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $9, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $10, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $11, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $12, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $13, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $14, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $15, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX2-NOPOPCNT-NEXT:    vpextrb $1, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    shrb %dl
; AVX2-NOPOPCNT-NEXT:    andb $85, %dl
; AVX2-NOPOPCNT-NEXT:    subb %dl, %cl
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    andb $51, %dl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    addb %dl, %cl
; AVX2-NOPOPCNT-NEXT:    movb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %dl
; AVX2-NOPOPCNT-NEXT:    addb %cl, %dl
; AVX2-NOPOPCNT-NEXT:    andb $15, %dl
; AVX2-NOPOPCNT-NEXT:    movzbl %dl, %ecx
; AVX2-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $2, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $3, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $4, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $5, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $6, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $7, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $8, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $9, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $10, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $11, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $12, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $13, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $14, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrb $15, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb %cl
; AVX2-NOPOPCNT-NEXT:    andb $85, %cl
; AVX2-NOPOPCNT-NEXT:    subb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $51, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $2, %al
; AVX2-NOPOPCNT-NEXT:    andb $51, %al
; AVX2-NOPOPCNT-NEXT:    addb %cl, %al
; AVX2-NOPOPCNT-NEXT:    movb %al, %cl
; AVX2-NOPOPCNT-NEXT:    shrb $4, %cl
; AVX2-NOPOPCNT-NEXT:    addb %al, %cl
; AVX2-NOPOPCNT-NEXT:    andb $15, %cl
; AVX2-NOPOPCNT-NEXT:    movzbl %cl, %eax
; AVX2-NOPOPCNT-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX2-NOPOPCNT-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    retq
  %out = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %in)
  ret <32 x i8> %out
}

define <4 x i64> @testv4i64(<4 x i64> %in) {
; AVX2-POPCNT-LABEL: testv4i64:
; AVX2-POPCNT:       # BB#0:
; AVX2-POPCNT-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-POPCNT-NEXT:    vpextrq $1, %xmm1, %rax
; AVX2-POPCNT-NEXT:    popcntq %rax, %rax
; AVX2-POPCNT-NEXT:    vmovq %rax, %xmm2
; AVX2-POPCNT-NEXT:    vmovq %xmm1, %rax
; AVX2-POPCNT-NEXT:    popcntq %rax, %rax
; AVX2-POPCNT-NEXT:    vmovq %rax, %xmm1
; AVX2-POPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX2-POPCNT-NEXT:    vpextrq $1, %xmm0, %rax
; AVX2-POPCNT-NEXT:    popcntq %rax, %rax
; AVX2-POPCNT-NEXT:    vmovq %rax, %xmm2
; AVX2-POPCNT-NEXT:    vmovq %xmm0, %rax
; AVX2-POPCNT-NEXT:    popcntq %rax, %rax
; AVX2-POPCNT-NEXT:    vmovq %rax, %xmm0
; AVX2-POPCNT-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX2-POPCNT-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-POPCNT-NEXT:    retq
;
; AVX2-NOPOPCNT-LABEL: testv4i64:
; AVX2-NOPOPCNT:       # BB#0:
; AVX2-NOPOPCNT-NEXT:    vpsrlq $1, %ymm0, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm2
; AVX2-NOPOPCNT-NEXT:    vpand %ymm2, %ymm1, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpsubq %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NOPOPCNT-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NOPOPCNT-NEXT:    vpsrlq $2, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpsrlq $4, %ymm0, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NOPOPCNT-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpsrlq $8, %ymm0, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpsrlq $16, %ymm0, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpsrlq $32, %ymm0, %ymm1
; AVX2-NOPOPCNT-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NOPOPCNT-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    retq
  %out = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %in)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) {
; AVX2-LABEL: testv8i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $1, %ymm0, %ymm1
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpsubd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vpsrld $2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpaddd %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vpsrld $4, %ymm0, %ymm1
; AVX2-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsrld $8, %ymm0, %ymm1
; AVX2-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsrld $16, %ymm0, %ymm1
; AVX2-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %in)
  ret <8 x i32> %out
}

define <16 x i16> @testv16i16(<16 x i16> %in) {
; AVX2-POPCNT-LABEL: testv16i16:
; AVX2-POPCNT:       # BB#0:
; AVX2-POPCNT-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-POPCNT-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX2-POPCNT-NEXT:    popcntw %cx, %cx
; AVX2-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-POPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $2, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $3, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $4, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $5, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $6, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $7, %xmm1, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX2-POPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX2-POPCNT-NEXT:    popcntw %cx, %cx
; AVX2-POPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-POPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-POPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX2-POPCNT-NEXT:    popcntw %ax, %ax
; AVX2-POPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX2-POPCNT-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-POPCNT-NEXT:    retq
;
; AVX2-NOPOPCNT-LABEL: testv16i16:
; AVX2-NOPOPCNT:       # BB#0:
; AVX2-NOPOPCNT-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NOPOPCNT-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vmovd %xmm1, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    shrl %edx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ch, %ecx # NOREX
; AVX2-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $2, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $3, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $4, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $5, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $6, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $7, %xmm1, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX2-NOPOPCNT-NEXT:    vpextrw $1, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vmovd %xmm0, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    shrl %edx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %edx, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %edx, %ecx
; AVX2-NOPOPCNT-NEXT:    movl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %edx
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %edx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ch, %ecx # NOREX
; AVX2-NOPOPCNT-NEXT:    vmovd %ecx, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $2, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $3, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $4, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $5, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $6, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NOPOPCNT-NEXT:    vpextrw $7, %xmm0, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    shrl %ecx
; AVX2-NOPOPCNT-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NOPOPCNT-NEXT:    subl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    shrl $2, %eax
; AVX2-NOPOPCNT-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NOPOPCNT-NEXT:    addl %ecx, %eax
; AVX2-NOPOPCNT-NEXT:    movl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NOPOPCNT-NEXT:    shrl $4, %ecx
; AVX2-NOPOPCNT-NEXT:    addl %eax, %ecx
; AVX2-NOPOPCNT-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NOPOPCNT-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NOPOPCNT-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NOPOPCNT-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX2-NOPOPCNT-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NOPOPCNT-NEXT:    retq
  %out = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %in)
  ret <16 x i16> %out
}

declare <32 x i8> @llvm.ctpop.v32i8(<32 x i8>)
declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>)
