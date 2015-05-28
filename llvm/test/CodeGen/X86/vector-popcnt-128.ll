; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2

target triple = "x86_64-unknown-unknown"

define <2 x i64> @testv2i64(<2 x i64> %in) {
; SSE-LABEL: testv2i64:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlq $1, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psubq %xmm1, %xmm0
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [3689348814741910323,3689348814741910323]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    psrlq $2, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    paddq %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlq $4, %xmm1
; SSE-NEXT:    paddq %xmm0, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    psrlq $8, %xmm0
; SSE-NEXT:    paddq %xmm1, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlq $16, %xmm1
; SSE-NEXT:    paddq %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    psrlq $32, %xmm0
; SSE-NEXT:    paddq %xmm1, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: testv2i64:
; AVX:       # BB#0:
; AVX-NEXT:    vpsrlq $1, %xmm0, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsubq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [3689348814741910323,3689348814741910323]
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vpsrlq $2, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpaddq %xmm0, %xmm2, %xmm0
; AVX-NEXT:    vpsrlq $4, %xmm0, %xmm1
; AVX-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpsrlq $8, %xmm0, %xmm1
; AVX-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlq $16, %xmm0, %xmm1
; AVX-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlq $32, %xmm0, %xmm1
; AVX-NEXT:    vpaddq %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    retq
  %out = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %in)
  ret <2 x i64> %out
}

define <4 x i32> @testv4i32(<4 x i32> %in) {
; SSE-LABEL: testv4i32:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrld $1, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psubd %xmm1, %xmm0
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [858993459,858993459,858993459,858993459]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    psrld $2, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    paddd %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrld $4, %xmm1
; SSE-NEXT:    paddd %xmm0, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm2
; SSE-NEXT:    psrld $8, %xmm2
; SSE-NEXT:    paddd %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    psrld $16, %xmm0
; SSE-NEXT:    paddd %xmm2, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    retq
;
; AVX1-LABEL: testv4i32:
; AVX1:       # BB#0:
; AVX1-NEXT:    vpsrld $1, %xmm0, %xmm1
; AVX1-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX1-NEXT:    vpsubd %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vmovdqa {{.*#+}} xmm1 = [858993459,858993459,858993459,858993459]
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX1-NEXT:    vpsrld $2, %xmm0, %xmm0
; AVX1-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; AVX1-NEXT:    vpsrld $4, %xmm0, %xmm1
; AVX1-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX1-NEXT:    vpsrld $8, %xmm0, %xmm1
; AVX1-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX1-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX1-NEXT:    retq
;
; AVX2-LABEL: testv4i32:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpsrld $1, %xmm0, %xmm1
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm2
; AVX2-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX2-NEXT:    vpsubd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX2-NEXT:    vpsrld $2, %xmm0, %xmm0
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpaddd %xmm0, %xmm2, %xmm0
; AVX2-NEXT:    vpsrld $4, %xmm0, %xmm1
; AVX2-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpsrld $8, %xmm0, %xmm1
; AVX2-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpsrld $16, %xmm0, %xmm1
; AVX2-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    retq
  %out = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %in)
  ret <4 x i32> %out
}

define <8 x i16> @testv8i16(<8 x i16> %in) {
; SSE2-LABEL: testv8i16:
; SSE2:       # BB#0:
; SSE2-NEXT:    pextrw $7, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    pextrw $3, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE2-NEXT:    pextrw $5, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    pextrw $1, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE2-NEXT:    pextrw $6, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    pextrw $2, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE2-NEXT:    pextrw $4, %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    movd %xmm0, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    shrl %ecx
; SSE2-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE2-NEXT:    subl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE2-NEXT:    shrl $2, %eax
; SSE2-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE2-NEXT:    addl %ecx, %eax
; SSE2-NEXT:    movl %eax, %ecx
; SSE2-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE2-NEXT:    shrl $4, %ecx
; SSE2-NEXT:    addl %eax, %ecx
; SSE2-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE2-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE2-NEXT:    movzbl %ah, %eax # NOREX
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv8i16:
; SSE3:       # BB#0:
; SSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    pextrw $3, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSE3-NEXT:    pextrw $5, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    pextrw $1, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSE3-NEXT:    pextrw $6, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    pextrw $2, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE3-NEXT:    pextrw $4, %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    movd %xmm0, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    shrl %ecx
; SSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE3-NEXT:    subl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE3-NEXT:    shrl $2, %eax
; SSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE3-NEXT:    addl %ecx, %eax
; SSE3-NEXT:    movl %eax, %ecx
; SSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE3-NEXT:    shrl $4, %ecx
; SSE3-NEXT:    addl %eax, %ecx
; SSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv8i16:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pextrw $7, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    pextrw $3, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3]
; SSSE3-NEXT:    pextrw $5, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    pextrw $1, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3]
; SSSE3-NEXT:    pextrw $6, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    pextrw $2, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSSE3-NEXT:    pextrw $4, %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    movd %xmm0, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    shrl %ecx
; SSSE3-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSSE3-NEXT:    subl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSSE3-NEXT:    shrl $2, %eax
; SSSE3-NEXT:    andl $13107, %eax # imm = 0x3333
; SSSE3-NEXT:    addl %ecx, %eax
; SSSE3-NEXT:    movl %eax, %ecx
; SSSE3-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSSE3-NEXT:    shrl $4, %ecx
; SSSE3-NEXT:    addl %eax, %ecx
; SSSE3-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSSE3-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSSE3-NEXT:    movzbl %ah, %eax # NOREX
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSSE3-NEXT:    punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv8i16:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrw $1, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    movd %xmm0, %ecx
; SSE41-NEXT:    movl %ecx, %edx
; SSE41-NEXT:    shrl %edx
; SSE41-NEXT:    andl $21845, %edx # imm = 0x5555
; SSE41-NEXT:    subl %edx, %ecx
; SSE41-NEXT:    movl %ecx, %edx
; SSE41-NEXT:    andl $13107, %edx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    addl %edx, %ecx
; SSE41-NEXT:    movl %ecx, %edx
; SSE41-NEXT:    andl $65520, %edx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %edx
; SSE41-NEXT:    addl %ecx, %edx
; SSE41-NEXT:    andl $3855, %edx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; SSE41-NEXT:    movzbl %ch, %ecx # NOREX
; SSE41-NEXT:    movd %ecx, %xmm1
; SSE41-NEXT:    pinsrw $1, %eax, %xmm1
; SSE41-NEXT:    pextrw $2, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $2, %eax, %xmm1
; SSE41-NEXT:    pextrw $3, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $3, %eax, %xmm1
; SSE41-NEXT:    pextrw $4, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $4, %eax, %xmm1
; SSE41-NEXT:    pextrw $5, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $5, %eax, %xmm1
; SSE41-NEXT:    pextrw $6, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $6, %eax, %xmm1
; SSE41-NEXT:    pextrw $7, %xmm0, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    shrl %ecx
; SSE41-NEXT:    andl $21845, %ecx # imm = 0x5555
; SSE41-NEXT:    subl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $13107, %ecx # imm = 0x3333
; SSE41-NEXT:    shrl $2, %eax
; SSE41-NEXT:    andl $13107, %eax # imm = 0x3333
; SSE41-NEXT:    addl %ecx, %eax
; SSE41-NEXT:    movl %eax, %ecx
; SSE41-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; SSE41-NEXT:    shrl $4, %ecx
; SSE41-NEXT:    addl %eax, %ecx
; SSE41-NEXT:    andl $3855, %ecx # imm = 0xF0F
; SSE41-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; SSE41-NEXT:    movzbl %ah, %eax # NOREX
; SSE41-NEXT:    pinsrw $7, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrw $1, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vmovd %xmm0, %ecx
; AVX-NEXT:    movl %ecx, %edx
; AVX-NEXT:    shrl %edx
; AVX-NEXT:    andl $21845, %edx # imm = 0x5555
; AVX-NEXT:    subl %edx, %ecx
; AVX-NEXT:    movl %ecx, %edx
; AVX-NEXT:    andl $13107, %edx # imm = 0x3333
; AVX-NEXT:    shrl $2, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    addl %edx, %ecx
; AVX-NEXT:    movl %ecx, %edx
; AVX-NEXT:    andl $65520, %edx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %edx
; AVX-NEXT:    addl %ecx, %edx
; AVX-NEXT:    andl $3855, %edx # imm = 0xF0F
; AVX-NEXT:    imull $257, %edx, %ecx # imm = 0x101
; AVX-NEXT:    movzbl %ch, %ecx # NOREX
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrw $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $2, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $3, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $3, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $4, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $4, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $5, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $5, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $6, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $6, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrw $7, %xmm0, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    shrl %ecx
; AVX-NEXT:    andl $21845, %ecx # imm = 0x5555
; AVX-NEXT:    subl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $13107, %ecx # imm = 0x3333
; AVX-NEXT:    shrl $2, %eax
; AVX-NEXT:    andl $13107, %eax # imm = 0x3333
; AVX-NEXT:    addl %ecx, %eax
; AVX-NEXT:    movl %eax, %ecx
; AVX-NEXT:    andl $65520, %ecx # imm = 0xFFF0
; AVX-NEXT:    shrl $4, %ecx
; AVX-NEXT:    addl %eax, %ecx
; AVX-NEXT:    andl $3855, %ecx # imm = 0xF0F
; AVX-NEXT:    imull $257, %ecx, %eax # imm = 0x101
; AVX-NEXT:    movzbl %ah, %eax # NOREX
; AVX-NEXT:    vpinsrw $7, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %in)
  ret <8 x i16> %out
}

define <16 x i8> @testv16i8(<16 x i8> %in) {
; SSE2-LABEL: testv16i8:
; SSE2:       # BB#0:
; SSE2-NEXT:    pushq %rbp
; SSE2-NEXT:  .Ltmp0:
; SSE2-NEXT:    .cfi_def_cfa_offset 16
; SSE2-NEXT:    pushq %rbx
; SSE2-NEXT:  .Ltmp1:
; SSE2-NEXT:    .cfi_def_cfa_offset 24
; SSE2-NEXT:  .Ltmp2:
; SSE2-NEXT:    .cfi_offset %rbx, -24
; SSE2-NEXT:  .Ltmp3:
; SSE2-NEXT:    .cfi_offset %rbp, -16
; SSE2-NEXT:    movaps %xmm0, -{{[0-9]+}}(%rsp)
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    addb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %al, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %r10b
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %dil
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %r9b
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %bpl
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %sil
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %bl
; SSE2-NEXT:    movb %bl, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %bl
; SSE2-NEXT:    movb %bl, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %bl
; SSE2-NEXT:    andb $51, %bl
; SSE2-NEXT:    addb %al, %bl
; SSE2-NEXT:    movb %bl, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %bl, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %dl
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %dl
; SSE2-NEXT:    andb $51, %dl
; SSE2-NEXT:    addb %al, %dl
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %dl, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %r11b
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %r8b
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE2-NEXT:    movb %al, %bl
; SSE2-NEXT:    shrb %bl
; SSE2-NEXT:    andb $85, %bl
; SSE2-NEXT:    subb %bl, %al
; SSE2-NEXT:    movb %al, %bl
; SSE2-NEXT:    andb $51, %bl
; SSE2-NEXT:    shrb $2, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    addb %bl, %al
; SSE2-NEXT:    movb %al, %bl
; SSE2-NEXT:    shrb $4, %bl
; SSE2-NEXT:    addb %al, %bl
; SSE2-NEXT:    andb $15, %bl
; SSE2-NEXT:    movzbl %bl, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    movb %cl, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %cl
; SSE2-NEXT:    movb %cl, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    addb %al, %cl
; SSE2-NEXT:    movb %cl, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %cl, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm1
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %dl
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %dl
; SSE2-NEXT:    andb $51, %dl
; SSE2-NEXT:    addb %al, %dl
; SSE2-NEXT:    movb %dl, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %dl, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:    movb %bpl, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %bpl
; SSE2-NEXT:    movb %bpl, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %bpl
; SSE2-NEXT:    andb $51, %bpl
; SSE2-NEXT:    addb %al, %bpl
; SSE2-NEXT:    movb %bpl, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %bpl, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSE2-NEXT:    movb %cl, %dl
; SSE2-NEXT:    shrb %dl
; SSE2-NEXT:    andb $85, %dl
; SSE2-NEXT:    subb %dl, %cl
; SSE2-NEXT:    movb %cl, %dl
; SSE2-NEXT:    andb $51, %dl
; SSE2-NEXT:    shrb $2, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    addb %dl, %cl
; SSE2-NEXT:    movb %cl, %dl
; SSE2-NEXT:    shrb $4, %dl
; SSE2-NEXT:    addb %cl, %dl
; SSE2-NEXT:    andb $15, %dl
; SSE2-NEXT:    movzbl %dl, %ecx
; SSE2-NEXT:    movd %ecx, %xmm1
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:    movb %r10b, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %r10b
; SSE2-NEXT:    movb %r10b, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %r10b
; SSE2-NEXT:    andb $51, %r10b
; SSE2-NEXT:    addb %cl, %r10b
; SSE2-NEXT:    movb %r10b, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %r10b, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %ecx
; SSE2-NEXT:    movd %ecx, %xmm2
; SSE2-NEXT:    movb %r11b, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %r11b
; SSE2-NEXT:    movb %r11b, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %r11b
; SSE2-NEXT:    andb $51, %r11b
; SSE2-NEXT:    addb %cl, %r11b
; SSE2-NEXT:    movb %r11b, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %r11b, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %ecx
; SSE2-NEXT:    movd %ecx, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    movb %r9b, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %r9b
; SSE2-NEXT:    movb %r9b, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %r9b
; SSE2-NEXT:    andb $51, %r9b
; SSE2-NEXT:    addb %cl, %r9b
; SSE2-NEXT:    movb %r9b, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %r9b, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %ecx
; SSE2-NEXT:    movd %ecx, %xmm3
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    addb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %al, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %eax
; SSE2-NEXT:    movd %eax, %xmm2
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:    movb %dil, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %dil
; SSE2-NEXT:    movb %dil, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %dil
; SSE2-NEXT:    andb $51, %dil
; SSE2-NEXT:    addb %al, %dil
; SSE2-NEXT:    movb %dil, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %dil, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    movb %r8b, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %r8b
; SSE2-NEXT:    movb %r8b, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %r8b
; SSE2-NEXT:    andb $51, %r8b
; SSE2-NEXT:    addb %al, %r8b
; SSE2-NEXT:    movb %r8b, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %r8b, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm3
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:    movb %sil, %al
; SSE2-NEXT:    shrb %al
; SSE2-NEXT:    andb $85, %al
; SSE2-NEXT:    subb %al, %sil
; SSE2-NEXT:    movb %sil, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    shrb $2, %sil
; SSE2-NEXT:    andb $51, %sil
; SSE2-NEXT:    addb %al, %sil
; SSE2-NEXT:    movb %sil, %al
; SSE2-NEXT:    shrb $4, %al
; SSE2-NEXT:    addb %sil, %al
; SSE2-NEXT:    andb $15, %al
; SSE2-NEXT:    movzbl %al, %eax
; SSE2-NEXT:    movd %eax, %xmm4
; SSE2-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb %cl
; SSE2-NEXT:    andb $85, %cl
; SSE2-NEXT:    subb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    andb $51, %cl
; SSE2-NEXT:    shrb $2, %al
; SSE2-NEXT:    andb $51, %al
; SSE2-NEXT:    addb %cl, %al
; SSE2-NEXT:    movb %al, %cl
; SSE2-NEXT:    shrb $4, %cl
; SSE2-NEXT:    addb %al, %cl
; SSE2-NEXT:    andb $15, %cl
; SSE2-NEXT:    movzbl %cl, %eax
; SSE2-NEXT:    movd %eax, %xmm0
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:    popq %rbx
; SSE2-NEXT:    popq %rbp
; SSE2-NEXT:    retq
;
; SSE3-LABEL: testv16i8:
; SSE3:       # BB#0:
; SSE3-NEXT:    pushq %rbp
; SSE3-NEXT:  .Ltmp0:
; SSE3-NEXT:    .cfi_def_cfa_offset 16
; SSE3-NEXT:    pushq %rbx
; SSE3-NEXT:  .Ltmp1:
; SSE3-NEXT:    .cfi_def_cfa_offset 24
; SSE3-NEXT:  .Ltmp2:
; SSE3-NEXT:    .cfi_offset %rbx, -24
; SSE3-NEXT:  .Ltmp3:
; SSE3-NEXT:    .cfi_offset %rbp, -16
; SSE3-NEXT:    movaps %xmm0, -{{[0-9]+}}(%rsp)
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    addb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %al, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r10b
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dil
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r9b
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %bpl
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %sil
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %bl
; SSE3-NEXT:    movb %bl, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %bl
; SSE3-NEXT:    movb %bl, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %bl
; SSE3-NEXT:    andb $51, %bl
; SSE3-NEXT:    addb %al, %bl
; SSE3-NEXT:    movb %bl, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %bl, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %dl
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %dl
; SSE3-NEXT:    andb $51, %dl
; SSE3-NEXT:    addb %al, %dl
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %dl, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r11b
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r8b
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE3-NEXT:    movb %al, %bl
; SSE3-NEXT:    shrb %bl
; SSE3-NEXT:    andb $85, %bl
; SSE3-NEXT:    subb %bl, %al
; SSE3-NEXT:    movb %al, %bl
; SSE3-NEXT:    andb $51, %bl
; SSE3-NEXT:    shrb $2, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    addb %bl, %al
; SSE3-NEXT:    movb %al, %bl
; SSE3-NEXT:    shrb $4, %bl
; SSE3-NEXT:    addb %al, %bl
; SSE3-NEXT:    andb $15, %bl
; SSE3-NEXT:    movzbl %bl, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE3-NEXT:    movb %cl, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %cl
; SSE3-NEXT:    movb %cl, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    addb %al, %cl
; SSE3-NEXT:    movb %cl, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %cl, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm1
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %dl
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %dl
; SSE3-NEXT:    andb $51, %dl
; SSE3-NEXT:    addb %al, %dl
; SSE3-NEXT:    movb %dl, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %dl, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE3-NEXT:    movb %bpl, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %bpl
; SSE3-NEXT:    movb %bpl, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %bpl
; SSE3-NEXT:    andb $51, %bpl
; SSE3-NEXT:    addb %al, %bpl
; SSE3-NEXT:    movb %bpl, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %bpl, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSE3-NEXT:    movb %cl, %dl
; SSE3-NEXT:    shrb %dl
; SSE3-NEXT:    andb $85, %dl
; SSE3-NEXT:    subb %dl, %cl
; SSE3-NEXT:    movb %cl, %dl
; SSE3-NEXT:    andb $51, %dl
; SSE3-NEXT:    shrb $2, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    addb %dl, %cl
; SSE3-NEXT:    movb %cl, %dl
; SSE3-NEXT:    shrb $4, %dl
; SSE3-NEXT:    addb %cl, %dl
; SSE3-NEXT:    andb $15, %dl
; SSE3-NEXT:    movzbl %dl, %ecx
; SSE3-NEXT:    movd %ecx, %xmm1
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE3-NEXT:    movb %r10b, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %r10b
; SSE3-NEXT:    movb %r10b, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %r10b
; SSE3-NEXT:    andb $51, %r10b
; SSE3-NEXT:    addb %cl, %r10b
; SSE3-NEXT:    movb %r10b, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %r10b, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %ecx
; SSE3-NEXT:    movd %ecx, %xmm2
; SSE3-NEXT:    movb %r11b, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %r11b
; SSE3-NEXT:    movb %r11b, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %r11b
; SSE3-NEXT:    andb $51, %r11b
; SSE3-NEXT:    addb %cl, %r11b
; SSE3-NEXT:    movb %r11b, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %r11b, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %ecx
; SSE3-NEXT:    movd %ecx, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    movb %r9b, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %r9b
; SSE3-NEXT:    movb %r9b, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %r9b
; SSE3-NEXT:    andb $51, %r9b
; SSE3-NEXT:    addb %cl, %r9b
; SSE3-NEXT:    movb %r9b, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %r9b, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %ecx
; SSE3-NEXT:    movd %ecx, %xmm3
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    addb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %al, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %eax
; SSE3-NEXT:    movd %eax, %xmm2
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE3-NEXT:    movb %dil, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %dil
; SSE3-NEXT:    movb %dil, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %dil
; SSE3-NEXT:    andb $51, %dil
; SSE3-NEXT:    addb %al, %dil
; SSE3-NEXT:    movb %dil, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %dil, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    movb %r8b, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %r8b
; SSE3-NEXT:    movb %r8b, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %r8b
; SSE3-NEXT:    andb $51, %r8b
; SSE3-NEXT:    addb %al, %r8b
; SSE3-NEXT:    movb %r8b, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %r8b, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm3
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE3-NEXT:    movb %sil, %al
; SSE3-NEXT:    shrb %al
; SSE3-NEXT:    andb $85, %al
; SSE3-NEXT:    subb %al, %sil
; SSE3-NEXT:    movb %sil, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    shrb $2, %sil
; SSE3-NEXT:    andb $51, %sil
; SSE3-NEXT:    addb %al, %sil
; SSE3-NEXT:    movb %sil, %al
; SSE3-NEXT:    shrb $4, %al
; SSE3-NEXT:    addb %sil, %al
; SSE3-NEXT:    andb $15, %al
; SSE3-NEXT:    movzbl %al, %eax
; SSE3-NEXT:    movd %eax, %xmm4
; SSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb %cl
; SSE3-NEXT:    andb $85, %cl
; SSE3-NEXT:    subb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    andb $51, %cl
; SSE3-NEXT:    shrb $2, %al
; SSE3-NEXT:    andb $51, %al
; SSE3-NEXT:    addb %cl, %al
; SSE3-NEXT:    movb %al, %cl
; SSE3-NEXT:    shrb $4, %cl
; SSE3-NEXT:    addb %al, %cl
; SSE3-NEXT:    andb $15, %cl
; SSE3-NEXT:    movzbl %cl, %eax
; SSE3-NEXT:    movd %eax, %xmm0
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE3-NEXT:    popq %rbx
; SSE3-NEXT:    popq %rbp
; SSE3-NEXT:    retq
;
; SSSE3-LABEL: testv16i8:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    pushq %rbp
; SSSE3-NEXT:  .Ltmp0:
; SSSE3-NEXT:    .cfi_def_cfa_offset 16
; SSSE3-NEXT:    pushq %rbx
; SSSE3-NEXT:  .Ltmp1:
; SSSE3-NEXT:    .cfi_def_cfa_offset 24
; SSSE3-NEXT:  .Ltmp2:
; SSSE3-NEXT:    .cfi_offset %rbx, -24
; SSSE3-NEXT:  .Ltmp3:
; SSSE3-NEXT:    .cfi_offset %rbp, -16
; SSSE3-NEXT:    movaps %xmm0, -{{[0-9]+}}(%rsp)
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    addb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %al, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r10b
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dil
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r9b
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %bpl
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %sil
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %bl
; SSSE3-NEXT:    movb %bl, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %bl
; SSSE3-NEXT:    movb %bl, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %bl
; SSSE3-NEXT:    andb $51, %bl
; SSSE3-NEXT:    addb %al, %bl
; SSSE3-NEXT:    movb %bl, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %bl, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %dl
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %dl
; SSSE3-NEXT:    andb $51, %dl
; SSSE3-NEXT:    addb %al, %dl
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %dl, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r11b
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %dl
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %r8b
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSSE3-NEXT:    movb %al, %bl
; SSSE3-NEXT:    shrb %bl
; SSSE3-NEXT:    andb $85, %bl
; SSSE3-NEXT:    subb %bl, %al
; SSSE3-NEXT:    movb %al, %bl
; SSSE3-NEXT:    andb $51, %bl
; SSSE3-NEXT:    shrb $2, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    addb %bl, %al
; SSSE3-NEXT:    movb %al, %bl
; SSSE3-NEXT:    shrb $4, %bl
; SSSE3-NEXT:    addb %al, %bl
; SSSE3-NEXT:    andb $15, %bl
; SSSE3-NEXT:    movzbl %bl, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    movb %cl, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %cl
; SSSE3-NEXT:    movb %cl, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    addb %al, %cl
; SSSE3-NEXT:    movb %cl, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %cl, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm1
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %dl
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %dl
; SSSE3-NEXT:    andb $51, %dl
; SSSE3-NEXT:    addb %al, %dl
; SSSE3-NEXT:    movb %dl, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %dl, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSSE3-NEXT:    movb %bpl, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %bpl
; SSSE3-NEXT:    movb %bpl, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %bpl
; SSSE3-NEXT:    andb $51, %bpl
; SSSE3-NEXT:    addb %al, %bpl
; SSSE3-NEXT:    movb %bpl, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %bpl, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %cl
; SSSE3-NEXT:    movb %cl, %dl
; SSSE3-NEXT:    shrb %dl
; SSSE3-NEXT:    andb $85, %dl
; SSSE3-NEXT:    subb %dl, %cl
; SSSE3-NEXT:    movb %cl, %dl
; SSSE3-NEXT:    andb $51, %dl
; SSSE3-NEXT:    shrb $2, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    addb %dl, %cl
; SSSE3-NEXT:    movb %cl, %dl
; SSSE3-NEXT:    shrb $4, %dl
; SSSE3-NEXT:    addb %cl, %dl
; SSSE3-NEXT:    andb $15, %dl
; SSSE3-NEXT:    movzbl %dl, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm1
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm3[0],xmm1[1],xmm3[1],xmm1[2],xmm3[2],xmm1[3],xmm3[3],xmm1[4],xmm3[4],xmm1[5],xmm3[5],xmm1[6],xmm3[6],xmm1[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSSE3-NEXT:    movb %r10b, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %r10b
; SSSE3-NEXT:    movb %r10b, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %r10b
; SSSE3-NEXT:    andb $51, %r10b
; SSSE3-NEXT:    addb %cl, %r10b
; SSSE3-NEXT:    movb %r10b, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %r10b, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm2
; SSSE3-NEXT:    movb %r11b, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %r11b
; SSSE3-NEXT:    movb %r11b, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %r11b
; SSSE3-NEXT:    andb $51, %r11b
; SSSE3-NEXT:    addb %cl, %r11b
; SSSE3-NEXT:    movb %r11b, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %r11b, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    movb %r9b, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %r9b
; SSSE3-NEXT:    movb %r9b, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %r9b
; SSSE3-NEXT:    andb $51, %r9b
; SSSE3-NEXT:    addb %cl, %r9b
; SSSE3-NEXT:    movb %r9b, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %r9b, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %ecx
; SSSE3-NEXT:    movd %ecx, %xmm3
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    addb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %al, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %eax
; SSSE3-NEXT:    movd %eax, %xmm2
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSSE3-NEXT:    movb %dil, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %dil
; SSSE3-NEXT:    movb %dil, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %dil
; SSSE3-NEXT:    andb $51, %dil
; SSSE3-NEXT:    addb %al, %dil
; SSSE3-NEXT:    movb %dil, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %dil, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    movb %r8b, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %r8b
; SSSE3-NEXT:    movb %r8b, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %r8b
; SSSE3-NEXT:    andb $51, %r8b
; SSSE3-NEXT:    addb %al, %r8b
; SSSE3-NEXT:    movb %r8b, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %r8b, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm3
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSSE3-NEXT:    movb %sil, %al
; SSSE3-NEXT:    shrb %al
; SSSE3-NEXT:    andb $85, %al
; SSSE3-NEXT:    subb %al, %sil
; SSSE3-NEXT:    movb %sil, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    shrb $2, %sil
; SSSE3-NEXT:    andb $51, %sil
; SSSE3-NEXT:    addb %al, %sil
; SSSE3-NEXT:    movb %sil, %al
; SSSE3-NEXT:    shrb $4, %al
; SSSE3-NEXT:    addb %sil, %al
; SSSE3-NEXT:    andb $15, %al
; SSSE3-NEXT:    movzbl %al, %eax
; SSSE3-NEXT:    movd %eax, %xmm4
; SSSE3-NEXT:    movb -{{[0-9]+}}(%rsp), %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb %cl
; SSSE3-NEXT:    andb $85, %cl
; SSSE3-NEXT:    subb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    andb $51, %cl
; SSSE3-NEXT:    shrb $2, %al
; SSSE3-NEXT:    andb $51, %al
; SSSE3-NEXT:    addb %cl, %al
; SSSE3-NEXT:    movb %al, %cl
; SSSE3-NEXT:    shrb $4, %cl
; SSSE3-NEXT:    addb %al, %cl
; SSSE3-NEXT:    andb $15, %cl
; SSSE3-NEXT:    movzbl %cl, %eax
; SSSE3-NEXT:    movd %eax, %xmm0
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSSE3-NEXT:    punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSSE3-NEXT:    popq %rbx
; SSSE3-NEXT:    popq %rbp
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: testv16i8:
; SSE41:       # BB#0:
; SSE41-NEXT:    pextrb $1, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pextrb $0, %xmm0, %ecx
; SSE41-NEXT:    movb %cl, %dl
; SSE41-NEXT:    shrb %dl
; SSE41-NEXT:    andb $85, %dl
; SSE41-NEXT:    subb %dl, %cl
; SSE41-NEXT:    movb %cl, %dl
; SSE41-NEXT:    andb $51, %dl
; SSE41-NEXT:    shrb $2, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    addb %dl, %cl
; SSE41-NEXT:    movb %cl, %dl
; SSE41-NEXT:    shrb $4, %dl
; SSE41-NEXT:    addb %cl, %dl
; SSE41-NEXT:    andb $15, %dl
; SSE41-NEXT:    movzbl %dl, %ecx
; SSE41-NEXT:    movd %ecx, %xmm1
; SSE41-NEXT:    pinsrb $1, %eax, %xmm1
; SSE41-NEXT:    pextrb $2, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $2, %eax, %xmm1
; SSE41-NEXT:    pextrb $3, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $3, %eax, %xmm1
; SSE41-NEXT:    pextrb $4, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $4, %eax, %xmm1
; SSE41-NEXT:    pextrb $5, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $5, %eax, %xmm1
; SSE41-NEXT:    pextrb $6, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $6, %eax, %xmm1
; SSE41-NEXT:    pextrb $7, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $7, %eax, %xmm1
; SSE41-NEXT:    pextrb $8, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $8, %eax, %xmm1
; SSE41-NEXT:    pextrb $9, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $9, %eax, %xmm1
; SSE41-NEXT:    pextrb $10, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $10, %eax, %xmm1
; SSE41-NEXT:    pextrb $11, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $11, %eax, %xmm1
; SSE41-NEXT:    pextrb $12, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $12, %eax, %xmm1
; SSE41-NEXT:    pextrb $13, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $13, %eax, %xmm1
; SSE41-NEXT:    pextrb $14, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $14, %eax, %xmm1
; SSE41-NEXT:    pextrb $15, %xmm0, %eax
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb %cl
; SSE41-NEXT:    andb $85, %cl
; SSE41-NEXT:    subb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    andb $51, %cl
; SSE41-NEXT:    shrb $2, %al
; SSE41-NEXT:    andb $51, %al
; SSE41-NEXT:    addb %cl, %al
; SSE41-NEXT:    movb %al, %cl
; SSE41-NEXT:    shrb $4, %cl
; SSE41-NEXT:    addb %al, %cl
; SSE41-NEXT:    andb $15, %cl
; SSE41-NEXT:    movzbl %cl, %eax
; SSE41-NEXT:    pinsrb $15, %eax, %xmm1
; SSE41-NEXT:    movdqa %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX-LABEL: testv16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vpextrb $1, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpextrb $0, %xmm0, %ecx
; AVX-NEXT:    movb %cl, %dl
; AVX-NEXT:    shrb %dl
; AVX-NEXT:    andb $85, %dl
; AVX-NEXT:    subb %dl, %cl
; AVX-NEXT:    movb %cl, %dl
; AVX-NEXT:    andb $51, %dl
; AVX-NEXT:    shrb $2, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    addb %dl, %cl
; AVX-NEXT:    movb %cl, %dl
; AVX-NEXT:    shrb $4, %dl
; AVX-NEXT:    addb %cl, %dl
; AVX-NEXT:    andb $15, %dl
; AVX-NEXT:    movzbl %dl, %ecx
; AVX-NEXT:    vmovd %ecx, %xmm1
; AVX-NEXT:    vpinsrb $1, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $2, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $2, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $3, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $3, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $4, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $4, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $5, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $5, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $6, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $6, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $7, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $7, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $8, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $8, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $9, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $9, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $10, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $10, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $11, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $11, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $12, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $12, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $13, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $13, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $14, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $14, %eax, %xmm1, %xmm1
; AVX-NEXT:    vpextrb $15, %xmm0, %eax
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb %cl
; AVX-NEXT:    andb $85, %cl
; AVX-NEXT:    subb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    andb $51, %cl
; AVX-NEXT:    shrb $2, %al
; AVX-NEXT:    andb $51, %al
; AVX-NEXT:    addb %cl, %al
; AVX-NEXT:    movb %al, %cl
; AVX-NEXT:    shrb $4, %cl
; AVX-NEXT:    addb %al, %cl
; AVX-NEXT:    andb $15, %cl
; AVX-NEXT:    movzbl %cl, %eax
; AVX-NEXT:    vpinsrb $15, %eax, %xmm1, %xmm0
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %in)
  ret <16 x i8> %out
}

declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>)
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>)
