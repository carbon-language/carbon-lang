; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX

;
; Vectorized integer shifts
;

define <2 x i64> @shl_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:
; SSE2:       pextrw $7, %xmm0, %eax
; SSE2-NEXT:  pextrw $7, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  pextrw $3, %xmm0, %eax
; SSE2-NEXT:  pextrw $3, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE2-NEXT:  pextrw $5, %xmm0, %eax
; SSE2-NEXT:  pextrw $5, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  pextrw $1, %xmm0, %eax
; SSE2-NEXT:  pextrw $1, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE2-NEXT:  pextrw $6, %xmm0, %eax
; SSE2-NEXT:  pextrw $6, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  pextrw $2, %xmm0, %eax
; SSE2-NEXT:  pextrw $2, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm4 = xmm4[0],xmm3[0],xmm4[1],xmm3[1],xmm4[2],xmm3[2],xmm4[3],xmm3[3]
; SSE2-NEXT:  pextrw $4, %xmm0, %eax
; SSE2-NEXT:  pextrw $4, %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  movd %xmm0, %eax
; SSE2-NEXT:  movd %xmm1, %ecx
; SSE2-NEXT:  shll %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:  retq
;
; SSE41:      pextrw $1, %xmm0, %eax
; SSE41-NEXT: pextrw $1, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: movd %xmm0, %edx
; SSE41-NEXT: movd %xmm1, %ecx
; SSE41-NEXT: shll %cl, %edx
; SSE41-NEXT: movd %edx, %xmm2
; SSE41-NEXT: pinsrw $1, %eax, %xmm2
; SSE41-NEXT: pextrw $2, %xmm0, %eax
; SSE41-NEXT: pextrw $2, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $2, %eax, %xmm2
; SSE41-NEXT: pextrw $3, %xmm0, %eax
; SSE41-NEXT: pextrw $3, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $3, %eax, %xmm2
; SSE41-NEXT: pextrw $4, %xmm0, %eax
; SSE41-NEXT: pextrw $4, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $4, %eax, %xmm2
; SSE41-NEXT: pextrw $5, %xmm0, %eax
; SSE41-NEXT: pextrw $5, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $5, %eax, %xmm2
; SSE41-NEXT: pextrw $6, %xmm0, %eax
; SSE41-NEXT: pextrw $6, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $6, %eax, %xmm2
; SSE41-NEXT: pextrw $7, %xmm0, %eax
; SSE41-NEXT: pextrw $7, %xmm1, %ecx
; SSE41-NEXT: shll %cl, %eax
; SSE41-NEXT: pinsrw $7, %eax, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpextrw $1, %xmm0, %eax
; AVX-NEXT:   vpextrw $1, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vmovd %xmm0, %edx
; AVX-NEXT:   vmovd %xmm1, %ecx
; AVX-NEXT:   shll %cl, %edx
; AVX-NEXT:   vmovd %edx, %xmm2
; AVX-NEXT:   vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $2, %xmm0, %eax
; AVX-NEXT:   vpextrw $2, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $3, %xmm0, %eax
; AVX-NEXT:   vpextrw $3, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $4, %xmm0, %eax
; AVX-NEXT:   vpextrw $4, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $5, %xmm0, %eax
; AVX-NEXT:   vpextrw $5, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $6, %xmm0, %eax
; AVX-NEXT:   vpextrw $6, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $7, %xmm0, %eax
; AVX-NEXT:   vpextrw $7, %xmm1, %ecx
; AVX-NEXT:   shll %cl, %eax
; AVX-NEXT:   vpinsrw $7, %eax, %xmm2, %xmm0
; AVX-NEXT:   retq
  %shl = shl <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %shl to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @shl_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; SSE2:       psllw $5, %xmm1
; SSE2-NEXT:  pand {{.*}}(%rip), %xmm1
; SSE2-NEXT:  movdqa {{.*#+}} xmm2 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; SSE2-NEXT:  movdqa %xmm2, %xmm3
; SSE2-NEXT:  pand %xmm1, %xmm3
; SSE2-NEXT:  pcmpeqb %xmm2, %xmm3
; SSE2-NEXT:  movdqa %xmm3, %xmm4
; SSE2-NEXT:  pandn %xmm0, %xmm4
; SSE2-NEXT:  psllw $4, %xmm0
; SSE2-NEXT:  pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand %xmm3, %xmm0
; SSE2-NEXT:  por %xmm4, %xmm0
; SSE2-NEXT:  paddb %xmm1, %xmm1
; SSE2-NEXT:  movdqa %xmm2, %xmm3
; SSE2-NEXT:  pand %xmm1, %xmm3
; SSE2-NEXT:  pcmpeqb %xmm2, %xmm3
; SSE2-NEXT:  movdqa %xmm3, %xmm4
; SSE2-NEXT:  pandn %xmm0, %xmm4
; SSE2-NEXT:  psllw $2, %xmm0
; SSE2-NEXT:  pand {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand %xmm3, %xmm0
; SSE2-NEXT:  por %xmm4, %xmm0
; SSE2-NEXT:  paddb %xmm1, %xmm1
; SSE2-NEXT:  pand %xmm2, %xmm1
; SSE2-NEXT:  pcmpeqb %xmm2, %xmm1
; SSE2-NEXT:  movdqa %xmm1, %xmm2
; SSE2-NEXT:  pandn %xmm0, %xmm2
; SSE2-NEXT:  paddb %xmm0, %xmm0
; SSE2-NEXT:  pand %xmm1, %xmm0
; SSE2-NEXT:  por %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa %xmm0, %xmm2
; SSE41-NEXT: psllw $5, %xmm1
; SSE41-NEXT: pand {{.*}}(%rip), %xmm1
; SSE41-NEXT: movdqa %xmm1, %xmm5
; SSE41-NEXT: paddb %xmm5, %xmm5
; SSE41-NEXT: movdqa {{.*#+}} xmm3 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; SSE41-NEXT: movdqa %xmm3, %xmm4
; SSE41-NEXT: pand %xmm5, %xmm4
; SSE41-NEXT: pcmpeqb %xmm3, %xmm4
; SSE41-NEXT: pand %xmm3, %xmm1
; SSE41-NEXT: pcmpeqb %xmm3, %xmm1
; SSE41-NEXT: movdqa %xmm2, %xmm6
; SSE41-NEXT: psllw $4, %xmm6
; SSE41-NEXT: pand {{.*}}(%rip), %xmm6
; SSE41-NEXT: movdqa %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm6, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm1
; SSE41-NEXT: psllw $2, %xmm1
; SSE41-NEXT: pand {{.*}}(%rip), %xmm1
; SSE41-NEXT: movdqa %xmm4, %xmm0
; SSE41-NEXT: pblendvb %xmm1, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm1
; SSE41-NEXT: paddb %xmm1, %xmm1
; SSE41-NEXT: paddb %xmm5, %xmm5
; SSE41-NEXT: pand %xmm3, %xmm5
; SSE41-NEXT: pcmpeqb %xmm5, %xmm3
; SSE41-NEXT: movdqa %xmm3, %xmm0
; SSE41-NEXT: pblendvb %xmm1, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw $5, %xmm1, %xmm1
; AVX-NEXT:   vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:   vpaddb %xmm1, %xmm1, %xmm2
; AVX-NEXT:   vmovdqa {{.*#+}} xmm3 = [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128]
; AVX-NEXT:   vpand %xmm2, %xmm3, %xmm4
; AVX-NEXT:   vpcmpeqb %xmm3, %xmm4, %xmm4
; AVX-NEXT:   vpand %xmm1, %xmm3, %xmm1
; AVX-NEXT:   vpcmpeqb %xmm3, %xmm1, %xmm1
; AVX-NEXT:   vpsllw $4, %xmm0, %xmm5
; AVX-NEXT:   vpand {{.*}}(%rip), %xmm5, %xmm5
; AVX-NEXT:   vpblendvb %xmm1, %xmm5, %xmm0, %xmm0
; AVX-NEXT:   vpsllw $2, %xmm0, %xmm1
; AVX-NEXT:   vpand {{.*}}(%rip), %xmm1, %xmm1
; AVX-NEXT:   vpblendvb %xmm4, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpaddb %xmm0, %xmm0, %xmm1
; AVX-NEXT:   vpaddb %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpand %xmm2, %xmm3, %xmm2
; AVX-NEXT:   vpcmpeqb %xmm3, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   retq
  %shl = shl <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %shl to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @ashr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:
; SSE2:       pextrw $7, %xmm1, %ecx
; SSE2-NEXT:  pextrw $7, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  pextrw $3, %xmm1, %ecx
; SSE2-NEXT:  pextrw $3, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE2-NEXT:  pextrw $5, %xmm1, %ecx
; SSE2-NEXT:  pextrw $5, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  pextrw $1, %xmm1, %ecx
; SSE2-NEXT:  pextrw $1, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE2-NEXT:  pextrw $6, %xmm1, %ecx
; SSE2-NEXT:  pextrw $6, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  pextrw $2, %xmm1, %ecx
; SSE2-NEXT:  pextrw $2, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm4 = xmm4[0],xmm3[0],xmm4[1],xmm3[1],xmm4[2],xmm3[2],xmm4[3],xmm3[3]
; SSE2-NEXT:  pextrw $4, %xmm1, %ecx
; SSE2-NEXT:  pextrw $4, %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  movd %xmm1, %ecx
; SSE2-NEXT:  movd %xmm0, %eax
; SSE2-NEXT:  sarw %cl, %ax
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:  retq
;
; SSE41:      pextrw $1, %xmm1, %ecx
; SSE41-NEXT: pextrw $1, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: movd %xmm1, %ecx
; SSE41-NEXT: movd %xmm0, %edx
; SSE41-NEXT: sarw %cl, %dx
; SSE41-NEXT: movd %edx, %xmm2
; SSE41-NEXT: pinsrw $1, %eax, %xmm2
; SSE41-NEXT: pextrw $2, %xmm1, %ecx
; SSE41-NEXT: pextrw $2, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $2, %eax, %xmm2
; SSE41-NEXT: pextrw $3, %xmm1, %ecx
; SSE41-NEXT: pextrw $3, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $3, %eax, %xmm2
; SSE41-NEXT: pextrw $4, %xmm1, %ecx
; SSE41-NEXT: pextrw $4, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $4, %eax, %xmm2
; SSE41-NEXT: pextrw $5, %xmm1, %ecx
; SSE41-NEXT: pextrw $5, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $5, %eax, %xmm2
; SSE41-NEXT: pextrw $6, %xmm1, %ecx
; SSE41-NEXT: pextrw $6, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $6, %eax, %xmm2
; SSE41-NEXT: pextrw $7, %xmm1, %ecx
; SSE41-NEXT: pextrw $7, %xmm0, %eax
; SSE41-NEXT: sarw %cl, %ax
; SSE41-NEXT: pinsrw $7, %eax, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpextrw $1, %xmm1, %ecx
; AVX-NEXT:   vpextrw $1, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vmovd %xmm1, %ecx
; AVX-NEXT:   vmovd %xmm0, %edx
; AVX-NEXT:   sarw %cl, %dx
; AVX-NEXT:   vmovd %edx, %xmm2
; AVX-NEXT:   vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $2, %xmm1, %ecx
; AVX-NEXT:   vpextrw $2, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $3, %xmm1, %ecx
; AVX-NEXT:   vpextrw $3, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $4, %xmm1, %ecx
; AVX-NEXT:   vpextrw $4, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $5, %xmm1, %ecx
; AVX-NEXT:   vpextrw $5, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $6, %xmm1, %ecx
; AVX-NEXT:   vpextrw $6, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $7, %xmm1, %ecx
; AVX-NEXT:   vpextrw $7, %xmm0, %eax
; AVX-NEXT:   sarw %cl, %ax
; AVX-NEXT:   vpinsrw $7, %eax, %xmm2, %xmm0
; AVX-NEXT:   retq
  %ashr = ashr <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %ashr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @ashr_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
;
; SSE2:       pushq %rbp
; SSE2-NEXT:  pushq %r15
; SSE2-NEXT:  pushq %r14
; SSE2-NEXT:  pushq %r13
; SSE2-NEXT:  pushq %r12
; SSE2-NEXT:  pushq %rbx
; SSE2-NEXT:  movaps %xmm1, -24(%rsp)
; SSE2-NEXT:  movaps %xmm0, -40(%rsp)
; SSE2-NEXT:  movb -9(%rsp), %cl
; SSE2-NEXT:  movb -25(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  movb -17(%rsp), %cl
; SSE2-NEXT:  movb -33(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -13(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movl %eax, -44(%rsp)
; SSE2-NEXT:  movb -29(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movzbl %al, %r9d
; SSE2-NEXT:  movb -21(%rsp), %cl
; SSE2-NEXT:  movb -37(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -11(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r10d
; SSE2-NEXT:  movb -27(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -19(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r11d
; SSE2-NEXT:  movb -35(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -15(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r14d
; SSE2-NEXT:  movb -31(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movzbl %al, %r15d
; SSE2-NEXT:  movb -23(%rsp), %cl
; SSE2-NEXT:  movb -39(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -10(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r12d
; SSE2-NEXT:  movb -26(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -18(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r13d
; SSE2-NEXT:  movb -34(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -14(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r8d
; SSE2-NEXT:  movb -30(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -22(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %ebp
; SSE2-NEXT:  movb -38(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movb -12(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %edi
; SSE2-NEXT:  movb -28(%rsp), %dl
; SSE2-NEXT:  sarb %cl, %dl
; SSE2-NEXT:  movb -20(%rsp), %cl
; SSE2-NEXT:  movzbl %dl, %esi
; SSE2-NEXT:  movb -36(%rsp), %bl
; SSE2-NEXT:  sarb %cl, %bl
; SSE2-NEXT:  movb -16(%rsp), %cl
; SSE2-NEXT:  movzbl %bl, %ebx
; SSE2-NEXT:  movb -32(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movzbl %al, %edx
; SSE2-NEXT:  movb -24(%rsp), %cl
; SSE2-NEXT:  movb -40(%rsp), %al
; SSE2-NEXT:  sarb %cl, %al
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movd -44(%rsp), %xmm1
; SSE2:       movd %r9d, %xmm2
; SSE2-NEXT:  movd %r10d, %xmm3
; SSE2-NEXT:  movd %r11d, %xmm4
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %r14d, %xmm0
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3],xmm3[4],xmm1[4],xmm3[5],xmm1[5],xmm3[6],xmm1[6],xmm3[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:  movd %r15d, %xmm1
; SSE2-NEXT:  movd %r12d, %xmm2
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE2-NEXT:  movd %r13d, %xmm0
; SSE2-NEXT:  movd %r8d, %xmm1
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %ebp, %xmm0
; SSE2-NEXT:  movd %edi, %xmm3
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3],xmm3[4],xmm1[4],xmm3[5],xmm1[5],xmm3[6],xmm1[6],xmm3[7],xmm1[7]
; SSE2-NEXT:  movd %esi, %xmm0
; SSE2-NEXT:  movd %ebx, %xmm1
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %edx, %xmm4
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:  popq %rbx
; SSE2-NEXT:  popq %r12
; SSE2-NEXT:  popq %r13
; SSE2-NEXT:  popq %r14
; SSE2-NEXT:  popq %r15
; SSE2-NEXT:  popq %rbp
; SSE2-NEXT:  retq
;
; SSE41:      pextrb $1, %xmm1, %ecx
; SSE41-NEXT: pextrb $1, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pextrb $0, %xmm1, %ecx
; SSE41-NEXT: pextrb $0, %xmm0, %edx
; SSE41-NEXT: sarb %cl, %dl
; SSE41-NEXT: movzbl %dl, %ecx
; SSE41-NEXT: movd %ecx, %xmm2
; SSE41-NEXT: pinsrb $1, %eax, %xmm2
; SSE41-NEXT: pextrb $2, %xmm1, %ecx
; SSE41-NEXT: pextrb $2, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $2, %eax, %xmm2
; SSE41-NEXT: pextrb $3, %xmm1, %ecx
; SSE41-NEXT: pextrb $3, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $3, %eax, %xmm2
; SSE41-NEXT: pextrb $4, %xmm1, %ecx
; SSE41-NEXT: pextrb $4, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $4, %eax, %xmm2
; SSE41-NEXT: pextrb $5, %xmm1, %ecx
; SSE41-NEXT: pextrb $5, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $5, %eax, %xmm2
; SSE41-NEXT: pextrb $6, %xmm1, %ecx
; SSE41-NEXT: pextrb $6, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $6, %eax, %xmm2
; SSE41-NEXT: pextrb $7, %xmm1, %ecx
; SSE41-NEXT: pextrb $7, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $7, %eax, %xmm2
; SSE41-NEXT: pextrb $8, %xmm1, %ecx
; SSE41-NEXT: pextrb $8, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $8, %eax, %xmm2
; SSE41-NEXT: pextrb $9, %xmm1, %ecx
; SSE41-NEXT: pextrb $9, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $9, %eax, %xmm2
; SSE41-NEXT: pextrb $10, %xmm1, %ecx
; SSE41-NEXT: pextrb $10, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $10, %eax, %xmm2
; SSE41-NEXT: pextrb $11, %xmm1, %ecx
; SSE41-NEXT: pextrb $11, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $11, %eax, %xmm2
; SSE41-NEXT: pextrb $12, %xmm1, %ecx
; SSE41-NEXT: pextrb $12, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $12, %eax, %xmm2
; SSE41-NEXT: pextrb $13, %xmm1, %ecx
; SSE41-NEXT: pextrb $13, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $13, %eax, %xmm2
; SSE41-NEXT: pextrb $14, %xmm1, %ecx
; SSE41-NEXT: pextrb $14, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $14, %eax, %xmm2
; SSE41-NEXT: pextrb $15, %xmm1, %ecx
; SSE41-NEXT: pextrb $15, %xmm0, %eax
; SSE41-NEXT: sarb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $15, %eax, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpextrb $1, %xmm1, %ecx
; AVX-NEXT:   vpextrb $1, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpextrb $0, %xmm1, %ecx
; AVX-NEXT:   vpextrb $0, %xmm0, %edx
; AVX-NEXT:   sarb %cl, %dl
; AVX-NEXT:   movzbl %dl, %ecx
; AVX-NEXT:   vmovd %ecx, %xmm2
; AVX-NEXT:   vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $2, %xmm1, %ecx
; AVX-NEXT:   vpextrb $2, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $3, %xmm1, %ecx
; AVX-NEXT:   vpextrb $3, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $4, %xmm1, %ecx
; AVX-NEXT:   vpextrb $4, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $5, %xmm1, %ecx
; AVX-NEXT:   vpextrb $5, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $6, %xmm1, %ecx
; AVX-NEXT:   vpextrb $6, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $7, %xmm1, %ecx
; AVX-NEXT:   vpextrb $7, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $8, %xmm1, %ecx
; AVX-NEXT:   vpextrb $8, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $9, %xmm1, %ecx
; AVX-NEXT:   vpextrb $9, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $10, %xmm1, %ecx
; AVX-NEXT:   vpextrb $10, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $11, %xmm1, %ecx
; AVX-NEXT:   vpextrb $11, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $12, %xmm1, %ecx
; AVX-NEXT:   vpextrb $12, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $13, %xmm1, %ecx
; AVX-NEXT:   vpextrb $13, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $14, %xmm1, %ecx
; AVX-NEXT:   vpextrb $14, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $15, %xmm1, %ecx
; AVX-NEXT:   vpextrb $15, %xmm0, %eax
; AVX-NEXT:   sarb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $15, %eax, %xmm2, %xmm0
; AVX-NEXT:   retq
  %ashr = ashr <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %ashr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @lshr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:

; SSE2:       pextrw $7, %xmm0, %eax
; SSE2-NEXT:  pextrw $7, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  pextrw $3, %xmm0, %eax
; SSE2-NEXT:  pextrw $3, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3]
; SSE2-NEXT:  pextrw $5, %xmm0, %eax
; SSE2-NEXT:  pextrw $5, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  pextrw $1, %xmm0, %eax
; SSE2-NEXT:  pextrw $1, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm2
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm4[0],xmm2[1],xmm4[1],xmm2[2],xmm4[2],xmm2[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3]
; SSE2-NEXT:  pextrw $6, %xmm0, %eax
; SSE2-NEXT:  pextrw $6, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  pextrw $2, %xmm0, %eax
; SSE2-NEXT:  pextrw $2, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm4
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm4 = xmm4[0],xmm3[0],xmm4[1],xmm3[1],xmm4[2],xmm3[2],xmm4[3],xmm3[3]
; SSE2-NEXT:  pextrw $4, %xmm0, %eax
; SSE2-NEXT:  pextrw $4, %xmm1, %ecx
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm3
; SSE2-NEXT:  movd %xmm1, %ecx
; SSE2-NEXT:  movd %xmm0, %eax
; SSE2-NEXT:  movzwl %ax, %eax
; SSE2-NEXT:  shrl %cl, %eax
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3]
; SSE2-NEXT:  punpcklwd {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3]
; SSE2-NEXT:  retq
;
; SSE41:      pextrw $1, %xmm0, %eax
; SSE41-NEXT: pextrw $1, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: movd %xmm1, %ecx
; SSE41-NEXT: movd %xmm0, %edx
; SSE41-NEXT: movzwl %dx, %edx
; SSE41-NEXT: shrl %cl, %edx
; SSE41-NEXT: movd %edx, %xmm2
; SSE41-NEXT: pinsrw $1, %eax, %xmm2
; SSE41-NEXT: pextrw $2, %xmm0, %eax
; SSE41-NEXT: pextrw $2, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $2, %eax, %xmm2
; SSE41-NEXT: pextrw $3, %xmm0, %eax
; SSE41-NEXT: pextrw $3, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $3, %eax, %xmm2
; SSE41-NEXT: pextrw $4, %xmm0, %eax
; SSE41-NEXT: pextrw $4, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $4, %eax, %xmm2
; SSE41-NEXT: pextrw $5, %xmm0, %eax
; SSE41-NEXT: pextrw $5, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $5, %eax, %xmm2
; SSE41-NEXT: pextrw $6, %xmm0, %eax
; SSE41-NEXT: pextrw $6, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $6, %eax, %xmm2
; SSE41-NEXT: pextrw $7, %xmm0, %eax
; SSE41-NEXT: pextrw $7, %xmm1, %ecx
; SSE41-NEXT: shrl %cl, %eax
; SSE41-NEXT: pinsrw $7, %eax, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpextrw $1, %xmm0, %eax
; AVX-NEXT:   vpextrw $1, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vmovd %xmm1, %ecx
; AVX-NEXT:   vmovd %xmm0, %edx
; AVX-NEXT:   movzwl %dx, %edx
; AVX-NEXT:   shrl %cl, %edx
; AVX-NEXT:   vmovd %edx, %xmm2
; AVX-NEXT:   vpinsrw $1, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $2, %xmm0, %eax
; AVX-NEXT:   vpextrw $2, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $2, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $3, %xmm0, %eax
; AVX-NEXT:   vpextrw $3, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $3, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $4, %xmm0, %eax
; AVX-NEXT:   vpextrw $4, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $4, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $5, %xmm0, %eax
; AVX-NEXT:   vpextrw $5, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $5, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $6, %xmm0, %eax
; AVX-NEXT:   vpextrw $6, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $6, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrw $7, %xmm0, %eax
; AVX-NEXT:   vpextrw $7, %xmm1, %ecx
; AVX-NEXT:   shrl %cl, %eax
; AVX-NEXT:   vpinsrw $7, %eax, %xmm2, %xmm0
; AVX-NEXT:   retq
  %lshr = lshr <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %lshr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @lshr_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; SSE2:       pushq %rbp
; SSE2-NEXT:  pushq %r15
; SSE2-NEXT:  pushq %r14
; SSE2-NEXT:  pushq %r13
; SSE2-NEXT:  pushq %r12
; SSE2-NEXT:  pushq %rbx
; SSE2-NEXT:  movaps %xmm1, -24(%rsp)
; SSE2-NEXT:  movaps %xmm0, -40(%rsp)
; SSE2-NEXT:  movb -9(%rsp), %cl
; SSE2-NEXT:  movb -25(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  movb -17(%rsp), %cl
; SSE2-NEXT:  movb -33(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -13(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movl %eax, -44(%rsp)
; SSE2-NEXT:  movb -29(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movzbl %al, %r9d
; SSE2-NEXT:  movb -21(%rsp), %cl
; SSE2-NEXT:  movb -37(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -11(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r10d
; SSE2-NEXT:  movb -27(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -19(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r11d
; SSE2-NEXT:  movb -35(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -15(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r14d
; SSE2-NEXT:  movb -31(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movzbl %al, %r15d
; SSE2-NEXT:  movb -23(%rsp), %cl
; SSE2-NEXT:  movb -39(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -10(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r12d
; SSE2-NEXT:  movb -26(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -18(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r13d
; SSE2-NEXT:  movb -34(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -14(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %r8d
; SSE2-NEXT:  movb -30(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -22(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %ebp
; SSE2-NEXT:  movb -38(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movb -12(%rsp), %cl
; SSE2-NEXT:  movzbl %al, %edi
; SSE2-NEXT:  movb -28(%rsp), %dl
; SSE2-NEXT:  shrb %cl, %dl
; SSE2-NEXT:  movb -20(%rsp), %cl
; SSE2-NEXT:  movzbl %dl, %esi
; SSE2-NEXT:  movb -36(%rsp), %bl
; SSE2-NEXT:  shrb %cl, %bl
; SSE2-NEXT:  movb -16(%rsp), %cl
; SSE2-NEXT:  movzbl %bl, %ebx
; SSE2-NEXT:  movb -32(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movzbl %al, %edx
; SSE2-NEXT:  movb -24(%rsp), %cl
; SSE2-NEXT:  movb -40(%rsp), %al
; SSE2-NEXT:  shrb %cl, %al
; SSE2-NEXT:  movzbl %al, %eax
; SSE2-NEXT:  movd -44(%rsp), %xmm1
; SSE2:       movd %r9d, %xmm2
; SSE2-NEXT:  movd %r10d, %xmm3
; SSE2-NEXT:  movd %r11d, %xmm4
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %r14d, %xmm0
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1],xmm3[2],xmm2[2],xmm3[3],xmm2[3],xmm3[4],xmm2[4],xmm3[5],xmm2[5],xmm3[6],xmm2[6],xmm3[7],xmm2[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3],xmm3[4],xmm1[4],xmm3[5],xmm1[5],xmm3[6],xmm1[6],xmm3[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:  movd %r15d, %xmm1
; SSE2-NEXT:  movd %r12d, %xmm2
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1],xmm2[2],xmm1[2],xmm2[3],xmm1[3],xmm2[4],xmm1[4],xmm2[5],xmm1[5],xmm2[6],xmm1[6],xmm2[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1],xmm2[2],xmm0[2],xmm2[3],xmm0[3],xmm2[4],xmm0[4],xmm2[5],xmm0[5],xmm2[6],xmm0[6],xmm2[7],xmm0[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1],xmm2[2],xmm3[2],xmm2[3],xmm3[3],xmm2[4],xmm3[4],xmm2[5],xmm3[5],xmm2[6],xmm3[6],xmm2[7],xmm3[7]
; SSE2-NEXT:  movd %r13d, %xmm0
; SSE2-NEXT:  movd %r8d, %xmm1
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %ebp, %xmm0
; SSE2-NEXT:  movd %edi, %xmm3
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm0[0],xmm3[1],xmm0[1],xmm3[2],xmm0[2],xmm3[3],xmm0[3],xmm3[4],xmm0[4],xmm3[5],xmm0[5],xmm3[6],xmm0[6],xmm3[7],xmm0[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm3 = xmm3[0],xmm1[0],xmm3[1],xmm1[1],xmm3[2],xmm1[2],xmm3[3],xmm1[3],xmm3[4],xmm1[4],xmm3[5],xmm1[5],xmm3[6],xmm1[6],xmm3[7],xmm1[7]
; SSE2-NEXT:  movd %esi, %xmm0
; SSE2-NEXT:  movd %ebx, %xmm1
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3],xmm1[4],xmm0[4],xmm1[5],xmm0[5],xmm1[6],xmm0[6],xmm1[7],xmm0[7]
; SSE2-NEXT:  movd %edx, %xmm4
; SSE2-NEXT:  movd %eax, %xmm0
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1],xmm0[2],xmm4[2],xmm0[3],xmm4[3],xmm0[4],xmm4[4],xmm0[5],xmm4[5],xmm0[6],xmm4[6],xmm0[7],xmm4[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm3[0],xmm0[1],xmm3[1],xmm0[2],xmm3[2],xmm0[3],xmm3[3],xmm0[4],xmm3[4],xmm0[5],xmm3[5],xmm0[6],xmm3[6],xmm0[7],xmm3[7]
; SSE2-NEXT:  punpcklbw {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1],xmm0[2],xmm2[2],xmm0[3],xmm2[3],xmm0[4],xmm2[4],xmm0[5],xmm2[5],xmm0[6],xmm2[6],xmm0[7],xmm2[7]
; SSE2-NEXT:  popq %rbx
; SSE2-NEXT:  popq %r12
; SSE2-NEXT:  popq %r13
; SSE2-NEXT:  popq %r14
; SSE2-NEXT:  popq %r15
; SSE2-NEXT:  popq %rbp
; SSE2-NEXT:  retq
;
; SSE41:      pextrb $1, %xmm1, %ecx
; SSE41-NEXT: pextrb $1, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pextrb $0, %xmm1, %ecx
; SSE41-NEXT: pextrb $0, %xmm0, %edx
; SSE41-NEXT: shrb %cl, %dl
; SSE41-NEXT: movzbl %dl, %ecx
; SSE41-NEXT: movd %ecx, %xmm2
; SSE41-NEXT: pinsrb $1, %eax, %xmm2
; SSE41-NEXT: pextrb $2, %xmm1, %ecx
; SSE41-NEXT: pextrb $2, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $2, %eax, %xmm2
; SSE41-NEXT: pextrb $3, %xmm1, %ecx
; SSE41-NEXT: pextrb $3, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $3, %eax, %xmm2
; SSE41-NEXT: pextrb $4, %xmm1, %ecx
; SSE41-NEXT: pextrb $4, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $4, %eax, %xmm2
; SSE41-NEXT: pextrb $5, %xmm1, %ecx
; SSE41-NEXT: pextrb $5, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $5, %eax, %xmm2
; SSE41-NEXT: pextrb $6, %xmm1, %ecx
; SSE41-NEXT: pextrb $6, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $6, %eax, %xmm2
; SSE41-NEXT: pextrb $7, %xmm1, %ecx
; SSE41-NEXT: pextrb $7, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $7, %eax, %xmm2
; SSE41-NEXT: pextrb $8, %xmm1, %ecx
; SSE41-NEXT: pextrb $8, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $8, %eax, %xmm2
; SSE41-NEXT: pextrb $9, %xmm1, %ecx
; SSE41-NEXT: pextrb $9, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $9, %eax, %xmm2
; SSE41-NEXT: pextrb $10, %xmm1, %ecx
; SSE41-NEXT: pextrb $10, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $10, %eax, %xmm2
; SSE41-NEXT: pextrb $11, %xmm1, %ecx
; SSE41-NEXT: pextrb $11, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $11, %eax, %xmm2
; SSE41-NEXT: pextrb $12, %xmm1, %ecx
; SSE41-NEXT: pextrb $12, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $12, %eax, %xmm2
; SSE41-NEXT: pextrb $13, %xmm1, %ecx
; SSE41-NEXT: pextrb $13, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $13, %eax, %xmm2
; SSE41-NEXT: pextrb $14, %xmm1, %ecx
; SSE41-NEXT: pextrb $14, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $14, %eax, %xmm2
; SSE41-NEXT: pextrb $15, %xmm1, %ecx
; SSE41-NEXT: pextrb $15, %xmm0, %eax
; SSE41-NEXT: shrb %cl, %al
; SSE41-NEXT: movzbl %al, %eax
; SSE41-NEXT: pinsrb $15, %eax, %xmm2
; SSE41-NEXT: movdqa %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpextrb $1, %xmm1, %ecx
; AVX-NEXT:   vpextrb $1, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpextrb $0, %xmm1, %ecx
; AVX-NEXT:   vpextrb $0, %xmm0, %edx
; AVX-NEXT:   shrb %cl, %dl
; AVX-NEXT:   movzbl %dl, %ecx
; AVX-NEXT:   vmovd %ecx, %xmm2
; AVX-NEXT:   vpinsrb $1, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $2, %xmm1, %ecx
; AVX-NEXT:   vpextrb $2, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $2, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $3, %xmm1, %ecx
; AVX-NEXT:   vpextrb $3, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $3, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $4, %xmm1, %ecx
; AVX-NEXT:   vpextrb $4, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $4, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $5, %xmm1, %ecx
; AVX-NEXT:   vpextrb $5, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $5, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $6, %xmm1, %ecx
; AVX-NEXT:   vpextrb $6, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $6, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $7, %xmm1, %ecx
; AVX-NEXT:   vpextrb $7, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $7, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $8, %xmm1, %ecx
; AVX-NEXT:   vpextrb $8, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $8, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $9, %xmm1, %ecx
; AVX-NEXT:   vpextrb $9, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $9, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $10, %xmm1, %ecx
; AVX-NEXT:   vpextrb $10, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $10, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $11, %xmm1, %ecx
; AVX-NEXT:   vpextrb $11, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $11, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $12, %xmm1, %ecx
; AVX-NEXT:   vpextrb $12, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $12, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $13, %xmm1, %ecx
; AVX-NEXT:   vpextrb $13, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $13, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $14, %xmm1, %ecx
; AVX-NEXT:   vpextrb $14, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $14, %eax, %xmm2, %xmm2
; AVX-NEXT:   vpextrb $15, %xmm1, %ecx
; AVX-NEXT:   vpextrb $15, %xmm0, %eax
; AVX-NEXT:   shrb %cl, %al
; AVX-NEXT:   movzbl %al, %eax
; AVX-NEXT:   vpinsrb $15, %eax, %xmm2, %xmm0
; AVX-NEXT:   retq
  %lshr = lshr <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %lshr to <2 x i64>
  ret <2 x i64> %tmp2
}
