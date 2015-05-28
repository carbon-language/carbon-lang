; RUN: llc < %s -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <4 x i64> @testv4i64(<4 x i64> %in) {
; AVX1-LABEL: testv4i64:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrq $1, %xmm1, %rdx
; AVX1-NEXT:    movq %rdx, %rax
; AVX1-NEXT:    shrq %rax
; AVX1-NEXT:    movabsq $6148914691236517205, %r8 # imm = 0x5555555555555555
; AVX1-NEXT:    andq %r8, %rax
; AVX1-NEXT:    subq %rax, %rdx
; AVX1-NEXT:    movabsq $3689348814741910323, %rax # imm = 0x3333333333333333
; AVX1-NEXT:    movq %rdx, %rsi
; AVX1-NEXT:    andq %rax, %rsi
; AVX1-NEXT:    shrq $2, %rdx
; AVX1-NEXT:    andq %rax, %rdx
; AVX1-NEXT:    addq %rsi, %rdx
; AVX1-NEXT:    movq %rdx, %rdi
; AVX1-NEXT:    shrq $4, %rdi
; AVX1-NEXT:    addq %rdx, %rdi
; AVX1-NEXT:    movabsq $1085102592571150095, %rdx # imm = 0xF0F0F0F0F0F0F0F
; AVX1-NEXT:    andq %rdx, %rdi
; AVX1-NEXT:    movabsq $72340172838076673, %rsi # imm = 0x101010101010101
; AVX1-NEXT:    imulq %rsi, %rdi
; AVX1-NEXT:    shrq $56, %rdi
; AVX1-NEXT:    vmovq %rdi, %xmm2
; AVX1-NEXT:    vmovq %xmm1, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    shrq %rdi
; AVX1-NEXT:    andq %r8, %rdi
; AVX1-NEXT:    subq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    andq %rax, %rdi
; AVX1-NEXT:    shrq $2, %rcx
; AVX1-NEXT:    andq %rax, %rcx
; AVX1-NEXT:    addq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    shrq $4, %rdi
; AVX1-NEXT:    addq %rcx, %rdi
; AVX1-NEXT:    andq %rdx, %rdi
; AVX1-NEXT:    imulq %rsi, %rdi
; AVX1-NEXT:    shrq $56, %rdi
; AVX1-NEXT:    vmovq %rdi, %xmm1
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm1 = xmm1[0],xmm2[0]
; AVX1-NEXT:    vpextrq $1, %xmm0, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    shrq %rdi
; AVX1-NEXT:    andq %r8, %rdi
; AVX1-NEXT:    subq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    andq %rax, %rdi
; AVX1-NEXT:    shrq $2, %rcx
; AVX1-NEXT:    andq %rax, %rcx
; AVX1-NEXT:    addq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    shrq $4, %rdi
; AVX1-NEXT:    addq %rcx, %rdi
; AVX1-NEXT:    andq %rdx, %rdi
; AVX1-NEXT:    imulq %rsi, %rdi
; AVX1-NEXT:    shrq $56, %rdi
; AVX1-NEXT:    vmovq %rdi, %xmm2
; AVX1-NEXT:    vmovq %xmm0, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    shrq %rdi
; AVX1-NEXT:    andq %r8, %rdi
; AVX1-NEXT:    subq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rdi
; AVX1-NEXT:    andq %rax, %rdi
; AVX1-NEXT:    shrq $2, %rcx
; AVX1-NEXT:    andq %rax, %rcx
; AVX1-NEXT:    addq %rdi, %rcx
; AVX1-NEXT:    movq %rcx, %rax
; AVX1-NEXT:    shrq $4, %rax
; AVX1-NEXT:    addq %rcx, %rax
; AVX1-NEXT:    andq %rdx, %rax
; AVX1-NEXT:    imulq %rsi, %rax
; AVX1-NEXT:    shrq $56, %rax
; AVX1-NEXT:    vmovq %rax, %xmm0
; AVX1-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm2[0]
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrlq $1, %ymm0, %ymm1
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm2
; AVX2-NEXT:    vpand %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpsubq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm2
; AVX2-NEXT:    vpsrlq $2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpaddq %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vpsrlq $4, %ymm0, %ymm1
; AVX2-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsrlq $8, %ymm0, %ymm1
; AVX2-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsrlq $16, %ymm0, %ymm1
; AVX2-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpsrlq $32, %ymm0, %ymm1
; AVX2-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; AVX2-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %in)
  ret <4 x i64> %out
}

define <8 x i32> @testv8i32(<8 x i32> %in) {
; AVX1-LABEL: testv8i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl %edx
; AVX1-NEXT:    andl $1431655765, %edx # imm = 0x55555555
; AVX1-NEXT:    subl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $858993459, %edx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    addl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl $4, %edx
; AVX1-NEXT:    addl %ecx, %edx
; AVX1-NEXT:    andl $252645135, %edx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %edx, %ecx # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrd $1, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl %edx
; AVX1-NEXT:    andl $1431655765, %edx # imm = 0x55555555
; AVX1-NEXT:    subl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $858993459, %edx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    addl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl $4, %edx
; AVX1-NEXT:    addl %ecx, %edx
; AVX1-NEXT:    andl $252645135, %edx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %edx, %ecx # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrd $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $2, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vpinsrd $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrd $3, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $1431655765, %ecx # imm = 0x55555555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $858993459, %ecx # imm = 0x33333333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $858993459, %eax # imm = 0x33333333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $252645135, %ecx # imm = 0xF0F0F0F
; AVX1-NEXT:    imull $16843009, %ecx, %eax # imm = 0x1010101
; AVX1-NEXT:    shrl $24, %eax
; AVX1-NEXT:    vpinsrd $3, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
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
; AVX1-LABEL: testv16i16:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vmovd %xmm1, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl %edx
; AVX1-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX1-NEXT:    subl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    addl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %edx
; AVX1-NEXT:    addl %ecx, %edx
; AVX1-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX1-NEXT:    movzbl %ch, %ecx # NOREX
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm1, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrw $1, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vmovd %xmm0, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    shrl %edx
; AVX1-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX1-NEXT:    subl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    addl %edx, %ecx
; AVX1-NEXT:    movl %ecx, %edx
; AVX1-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %edx
; AVX1-NEXT:    addl %ecx, %edx
; AVX1-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX1-NEXT:    movzbl %ch, %ecx # NOREX
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $2, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $3, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $4, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $5, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $6, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrw $7, %xmm0, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    shrl %ecx
; AVX1-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX1-NEXT:    subl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX1-NEXT:    shrl $2, %eax
; AVX1-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX1-NEXT:    addl %ecx, %eax
; AVX1-NEXT:    movl %eax, %ecx
; AVX1-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX1-NEXT:    shrl $4, %ecx
; AVX1-NEXT:    addl %eax, %ecx
; AVX1-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX1-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX1-NEXT:    movzbl %ah, %eax # NOREX
; AVX1-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv16i16:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vmovd %xmm1, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    shrl %edx
; AVX2-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX2-NEXT:    subl %edx, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    addl %edx, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %edx
; AVX2-NEXT:    addl %ecx, %edx
; AVX2-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX2-NEXT:    movzbl %ch, %ecx # NOREX
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm1, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrw $1, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vmovd %xmm0, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    shrl %edx
; AVX2-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX2-NEXT:    subl %edx, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    addl %edx, %ecx
; AVX2-NEXT:    movl %ecx, %edx
; AVX2-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %edx
; AVX2-NEXT:    addl %ecx, %edx
; AVX2-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX2-NEXT:    movzbl %ch, %ecx # NOREX
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrw $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $2, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $3, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $4, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $5, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $6, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrw $7, %xmm0, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    shrl %ecx
; AVX2-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX2-NEXT:    subl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX2-NEXT:    shrl $2, %eax
; AVX2-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX2-NEXT:    addl %ecx, %eax
; AVX2-NEXT:    movl %eax, %ecx
; AVX2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX2-NEXT:    shrl $4, %ecx
; AVX2-NEXT:    addl %eax, %ecx
; AVX2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX2-NEXT:    movzbl %ah, %eax # NOREX
; AVX2-NEXT:    vpinsrw $7, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %in)
  ret <16 x i16> %out
}

define <32 x i8> @testv32i8(<32 x i8> %in) {
; AVX1-LABEL: testv32i8:
; AVX1:       # BB#0:
; AVX1-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    shrb %dl
; AVX1-NEXT:    andb $85, %dl
; AVX1-NEXT:    subb %dl, %cl
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    andb $51, %dl
; AVX1-NEXT:    shrb $2, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    addb %dl, %cl
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    shrb $4, %dl
; AVX1-NEXT:    addb %cl, %dl
; AVX1-NEXT:    andb $15, %dl
; AVX1-NEXT:    movzbl %dl, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm1, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX1-NEXT:    vpextrb $1, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    shrb %dl
; AVX1-NEXT:    andb $85, %dl
; AVX1-NEXT:    subb %dl, %cl
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    andb $51, %dl
; AVX1-NEXT:    shrb $2, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    addb %dl, %cl
; AVX1-NEXT:    movb %cl, %dl
; AVX1-NEXT:    shrb $4, %dl
; AVX1-NEXT:    addb %cl, %dl
; AVX1-NEXT:    andb $15, %dl
; AVX1-NEXT:    movzbl %dl, %ecx
; AVX1-NEXT:    vmovd %ecx, %xmm2
; AVX1-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $2, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $3, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $4, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $5, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $6, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $7, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $8, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $9, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $10, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $11, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $12, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $13, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $14, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX1-NEXT:    vpextrb $15, %xmm0, %eax
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb %cl
; AVX1-NEXT:    andb $85, %cl
; AVX1-NEXT:    subb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    andb $51, %cl
; AVX1-NEXT:    shrb $2, %al
; AVX1-NEXT:    andb $51, %al
; AVX1-NEXT:    addb %cl, %al
; AVX1-NEXT:    movb %al, %cl
; AVX1-NEXT:    shrb $4, %cl
; AVX1-NEXT:    addb %al, %cl
; AVX1-NEXT:    andb $15, %cl
; AVX1-NEXT:    movzbl %cl, %eax
; AVX1-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX1-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv32i8:
; AVX2:       # BB#0:
; AVX2-NEXT:    vextracti128 $1, %ymm0, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpextrb $0, %xmm1, %ecx
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    shrb %dl
; AVX2-NEXT:    andb $85, %dl
; AVX2-NEXT:    subb %dl, %cl
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    andb $51, %dl
; AVX2-NEXT:    shrb $2, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    addb %dl, %cl
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    shrb $4, %dl
; AVX2-NEXT:    addb %cl, %dl
; AVX2-NEXT:    andb $15, %dl
; AVX2-NEXT:    movzbl %dl, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm1, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm1
; AVX2-NEXT:    vpextrb $1, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    shrb %dl
; AVX2-NEXT:    andb $85, %dl
; AVX2-NEXT:    subb %dl, %cl
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    andb $51, %dl
; AVX2-NEXT:    shrb $2, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    addb %dl, %cl
; AVX2-NEXT:    movb %cl, %dl
; AVX2-NEXT:    shrb $4, %dl
; AVX2-NEXT:    addb %cl, %dl
; AVX2-NEXT:    andb $15, %dl
; AVX2-NEXT:    movzbl %dl, %ecx
; AVX2-NEXT:    vmovd %ecx, %xmm2
; AVX2-NEXT:    vpinsrb $1, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $2, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $2, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $3, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $3, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $4, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $4, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $5, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $5, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $6, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $6, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $7, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $7, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $8, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $8, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $9, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $9, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $10, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $10, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $11, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $11, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $12, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $12, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $13, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $13, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $14, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $14, %eax, %xmm2, %xmm2
; AVX2-NEXT:    vpextrb $15, %xmm0, %eax
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb %cl
; AVX2-NEXT:    andb $85, %cl
; AVX2-NEXT:    subb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    andb $51, %cl
; AVX2-NEXT:    shrb $2, %al
; AVX2-NEXT:    andb $51, %al
; AVX2-NEXT:    addb %cl, %al
; AVX2-NEXT:    movb %al, %cl
; AVX2-NEXT:    shrb $4, %cl
; AVX2-NEXT:    addb %al, %cl
; AVX2-NEXT:    andb $15, %cl
; AVX2-NEXT:    movzbl %cl, %eax
; AVX2-NEXT:    vpinsrb $15, %eax, %xmm2, %xmm0
; AVX2-NEXT:    vinserti128 $1, %xmm1, %ymm0, %ymm0
; AVX2-NEXT:    retq
  %out = call <32 x i8> @llvm.ctpop.v32i8(<32 x i8> %in)
  ret <32 x i8> %out
}

declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>)
declare <32 x i8> @llvm.ctpop.v32i8(<32 x i8>)
