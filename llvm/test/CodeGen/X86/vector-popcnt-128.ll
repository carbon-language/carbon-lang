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
; SSE-NEXT:    psllq $32, %xmm0
; SSE-NEXT:    paddb %xmm1, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psllq $16, %xmm1
; SSE-NEXT:    paddb %xmm0, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    psllq $8, %xmm0
; SSE-NEXT:    paddb %xmm1, %xmm0
; SSE-NEXT:    psrlq $56, %xmm0
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
; AVX-NEXT:    vpsllq $32, %xmm0, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsllq $16, %xmm0, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsllq $8, %xmm0, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlq $56, %xmm0, %xmm0
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
; SSE-NEXT:    psllq $16, %xmm2
; SSE-NEXT:    paddb %xmm1, %xmm2
; SSE-NEXT:    movdqa %xmm2, %xmm0
; SSE-NEXT:    psllq $8, %xmm0
; SSE-NEXT:    paddb %xmm2, %xmm0
; SSE-NEXT:    psrld $24, %xmm0
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
; AVX1-NEXT:    vpsllq $16, %xmm0, %xmm1
; AVX1-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpsllq $8, %xmm0, %xmm1
; AVX1-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vpsrld $24, %xmm0, %xmm0
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
; AVX2-NEXT:    vpsllq $16, %xmm0, %xmm1
; AVX2-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpsllq $8, %xmm0, %xmm1
; AVX2-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpsrld $24, %xmm0, %xmm0
; AVX2-NEXT:    retq
  %out = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %in)
  ret <4 x i32> %out
}

define <8 x i16> @testv8i16(<8 x i16> %in) {
; SSE-LABEL: testv8i16:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlw $1, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psubw %xmm1, %xmm0
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [13107,13107,13107,13107,13107,13107,13107,13107]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    psrlw $2, %xmm0
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    paddw %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlw $4, %xmm1
; SSE-NEXT:    paddw %xmm0, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    psllq $8, %xmm0
; SSE-NEXT:    paddb %xmm1, %xmm0
; SSE-NEXT:    psrlw $8, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: testv8i16:
; AVX:       # BB#0:
; AVX-NEXT:    vpsrlw $1, %xmm0, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsubw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [13107,13107,13107,13107,13107,13107,13107,13107]
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vpsrlw $2, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpaddw %xmm0, %xmm2, %xmm0
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm1
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpsllq $8, %xmm0, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpsrlw $8, %xmm0, %xmm0
; AVX-NEXT:    retq
  %out = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %in)
  ret <8 x i16> %out
}

define <16 x i8> @testv16i8(<16 x i8> %in) {
; SSE-LABEL: testv16i8:
; SSE:       # BB#0:
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlw $1, %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    pand {{.*}}(%rip), %xmm1
; SSE-NEXT:    psubb %xmm1, %xmm0
; SSE-NEXT:    movdqa {{.*#+}} xmm1 = [51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51]
; SSE-NEXT:    movdqa %xmm0, %xmm2
; SSE-NEXT:    pand %xmm1, %xmm2
; SSE-NEXT:    psrlw $2, %xmm0
; SSE-NEXT:    pand {{.*}}(%rip), %xmm0
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    paddb %xmm2, %xmm0
; SSE-NEXT:    movdqa %xmm0, %xmm1
; SSE-NEXT:    psrlw $4, %xmm1
; SSE-NEXT:    movdqa {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; SSE-NEXT:    pand %xmm2, %xmm1
; SSE-NEXT:    paddb %xmm0, %xmm1
; SSE-NEXT:    pand %xmm2, %xmm1
; SSE-NEXT:    movdqa %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: testv16i8:
; AVX:       # BB#0:
; AVX-NEXT:    vpsrlw $1, %xmm0, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:    vpsubb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vmovdqa {{.*#+}} xmm1 = [51,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51]
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm2
; AVX-NEXT:    vpsrlw $2, %xmm0, %xmm0
; AVX-NEXT:    vpand {{.*}}(%rip), %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpaddb %xmm0, %xmm2, %xmm0
; AVX-NEXT:    vpsrlw $4, %xmm0, %xmm1
; AVX-NEXT:    vmovdqa {{.*#+}} xmm2 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
; AVX-NEXT:    vpand %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm2, %xmm0, %xmm0
; AVX-NEXT:    retq
  %out = call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %in)
  ret <16 x i8> %out
}

declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>)
declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>)
