; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 | FileCheck %s --check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX

;
; Vectorized integer shifts
;

define <2 x i64> @shl_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:
; ALL-NOT: shll
;
; SSE2:       psllw   $12, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psllw   $8, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psllw   $4, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psllw   $2, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  psraw   $15, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  pandn   %xmm0, %xmm2
; SSE2-NEXT:  psllw   $1, %xmm0
; SSE2-NEXT:  pand    %xmm1, %xmm0
; SSE2-NEXT:  por     %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa   %xmm0, %xmm2
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: psllw    $12, %xmm0
; SSE41-NEXT: psllw    $4, %xmm1
; SSE41-NEXT: por      %xmm0, %xmm1
; SSE41-NEXT: movdqa   %xmm1, %xmm3
; SSE41-NEXT: paddw    %xmm3, %xmm3
; SSE41-NEXT: movdqa   %xmm2, %xmm4
; SSE41-NEXT: psllw    $8, %xmm4
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm4, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm1
; SSE41-NEXT: psllw    $4, %xmm1
; SSE41-NEXT: movdqa   %xmm3, %xmm0
; SSE41-NEXT: pblendvb %xmm1, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm1
; SSE41-NEXT: psllw    $2, %xmm1
; SSE41-NEXT: paddw    %xmm3, %xmm3
; SSE41-NEXT: movdqa   %xmm3, %xmm0
; SSE41-NEXT: pblendvb %xmm1, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm1
; SSE41-NEXT: psllw    $1, %xmm1
; SSE41-NEXT: paddw    %xmm3, %xmm3
; SSE41-NEXT: movdqa   %xmm3, %xmm0
; SSE41-NEXT: pblendvb %xmm1, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw    $12, %xmm1, %xmm2
; AVX-NEXT:   vpsllw    $4, %xmm1, %xmm1
; AVX-NEXT:   vpor      %xmm2, %xmm1, %xmm1
; AVX-NEXT:   vpaddw    %xmm1, %xmm1, %xmm2
; AVX-NEXT:   vpsllw    $8, %xmm0, %xmm3
; AVX-NEXT:   vpblendvb %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsllw    $4, %xmm0, %xmm1
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsllw    $2, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsllw    $1, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   retq
  %shl = shl <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %shl to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @shl_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; SSE2:       psllw   $5, %xmm1
; SSE2-NEXT:  pxor    %xmm2, %xmm2
; SSE2-NEXT:  pxor    %xmm3, %xmm3
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm3
; SSE2-NEXT:  movdqa  %xmm3, %xmm4
; SSE2-NEXT:  pandn   %xmm0, %xmm4
; SSE2-NEXT:  psllw   $4, %xmm0
; SSE2-NEXT:  pand    {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand    %xmm3, %xmm0
; SSE2-NEXT:  por     %xmm4, %xmm0
; SSE2-NEXT:  paddb   %xmm1, %xmm1
; SSE2-NEXT:  pxor    %xmm3, %xmm3
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm3
; SSE2-NEXT:  movdqa  %xmm3, %xmm4
; SSE2-NEXT:  pandn   %xmm0, %xmm4
; SSE2-NEXT:  psllw   $2, %xmm0
; SSE2-NEXT:  pand    {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand    %xmm3, %xmm0
; SSE2-NEXT:  por     %xmm4, %xmm0
; SSE2-NEXT:  paddb   %xmm1, %xmm1
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm1
; SSE2-NEXT:  pandn   %xmm0, %xmm1
; SSE2-NEXT:  paddb   %xmm0, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm1, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa   %xmm0, %xmm2
; SSE41-NEXT: psllw    $5, %xmm1
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: psllw    $4, %xmm3
; SSE41-NEXT: pand     {{.*}}(%rip), %xmm3
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: psllw    $2, %xmm3
; SSE41-NEXT: pand     {{.*}}(%rip), %xmm3
; SSE41-NEXT: paddb    %xmm1, %xmm1
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: paddb    %xmm3, %xmm3
; SSE41-NEXT: paddb    %xmm1, %xmm1
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw    $5, %xmm1, %xmm1
; AVX-NEXT:   vpsllw    $4, %xmm0, %xmm2
; AVX-NEXT:   vpand     {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   vpsllw    $2, %xmm0, %xmm2
; AVX-NEXT:   vpand     {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:   vpaddb    %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   vpaddb    %xmm0, %xmm0, %xmm2
; AVX-NEXT:   vpaddb    %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   retq
  %shl = shl <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %shl to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @ashr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:
; ALL-NOT: sarw
;
; SSE2:       psllw   $12, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psraw   $8, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psraw   $4, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psraw   $2, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  psraw   $15, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  pandn   %xmm0, %xmm2
; SSE2-NEXT:  psraw   $1, %xmm0
; SSE2-NEXT:  pand    %xmm1, %xmm0
; SSE2-NEXT:  por     %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa    %xmm0, %xmm2
; SSE41-NEXT: movdqa    %xmm1, %xmm0
; SSE41-NEXT: psllw     $12, %xmm0
; SSE41-NEXT: psllw     $4, %xmm1
; SSE41-NEXT: por       %xmm0, %xmm1
; SSE41-NEXT: movdqa    %xmm1, %xmm3
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm2, %xmm4
; SSE41-NEXT: psraw     $8, %xmm4
; SSE41-NEXT: movdqa    %xmm1, %xmm0
; SSE41-NEXT: pblendvb  %xmm4, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psraw     $4, %xmm1
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psraw     $2, %xmm1
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psraw     $1, %xmm1
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw    $12, %xmm1, %xmm2
; AVX-NEXT:   vpsllw    $4, %xmm1, %xmm1
; AVX-NEXT:   vpor      %xmm2, %xmm1, %xmm1
; AVX-NEXT:   vpaddw    %xmm1, %xmm1, %xmm2
; AVX-NEXT:   vpsraw    $8, %xmm0, %xmm3
; AVX-NEXT:   vpblendvb %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsraw    $4, %xmm0, %xmm1
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsraw    $2, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsraw    $1, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   retq
  %ashr = ashr <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %ashr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @ashr_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; ALL-NOT: sarb
;
; SSE2:       punpckhbw {{.*#}} xmm2 = xmm2[8],xmm0[8],xmm2[9],xmm0[9],xmm2[10],xmm0[10],xmm2[11],xmm0[11],xmm2[12],xmm0[12],xmm2[13],xmm0[13],xmm2[14],xmm0[14],xmm2[15],xmm0[15]
; SSE2-NEXT:  psllw     $5, %xmm1
; SSE2-NEXT:  punpckhbw {{.*#}} xmm4 = xmm4[8],xmm1[8],xmm4[9],xmm1[9],xmm4[10],xmm1[10],xmm4[11],xmm1[11],xmm4[12],xmm1[12],xmm4[13],xmm1[13],xmm4[14],xmm1[14],xmm4[15],xmm1[15]
; SSE2-NEXT:  pxor      %xmm3, %xmm3
; SSE2-NEXT:  pxor      %xmm5, %xmm5
; SSE2-NEXT:  pcmpgtw   %xmm4, %xmm5
; SSE2-NEXT:  movdqa    %xmm5, %xmm6
; SSE2-NEXT:  pandn     %xmm2, %xmm6
; SSE2-NEXT:  psraw     $4, %xmm2
; SSE2-NEXT:  pand      %xmm5, %xmm2
; SSE2-NEXT:  por       %xmm6, %xmm2
; SSE2-NEXT:  paddw     %xmm4, %xmm4
; SSE2-NEXT:  pxor      %xmm5, %xmm5
; SSE2-NEXT:  pcmpgtw   %xmm4, %xmm5
; SSE2-NEXT:  movdqa    %xmm5, %xmm6
; SSE2-NEXT:  pandn     %xmm2, %xmm6
; SSE2-NEXT:  psraw     $2, %xmm2
; SSE2-NEXT:  pand      %xmm5, %xmm2
; SSE2-NEXT:  por       %xmm6, %xmm2
; SSE2-NEXT:  paddw     %xmm4, %xmm4
; SSE2-NEXT:  pxor      %xmm5, %xmm5
; SSE2-NEXT:  pcmpgtw   %xmm4, %xmm5
; SSE2-NEXT:  movdqa    %xmm5, %xmm4
; SSE2-NEXT:  pandn     %xmm2, %xmm4
; SSE2-NEXT:  psraw     $1, %xmm2
; SSE2-NEXT:  pand      %xmm5, %xmm2
; SSE2-NEXT:  por       %xmm4, %xmm2
; SSE2-NEXT:  psrlw     $8, %xmm2
; SSE2-NEXT:  punpcklbw {{.*#}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:  punpcklbw {{.*#}} xmm1 = xmm1[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; SSE2-NEXT:  pxor      %xmm4, %xmm4
; SSE2-NEXT:  pcmpgtw   %xmm1, %xmm4
; SSE2-NEXT:  movdqa    %xmm4, %xmm5
; SSE2-NEXT:  pandn     %xmm0, %xmm5
; SSE2-NEXT:  psraw     $4, %xmm0
; SSE2-NEXT:  pand      %xmm4, %xmm0
; SSE2-NEXT:  por       %xmm5, %xmm0
; SSE2-NEXT:  paddw     %xmm1, %xmm1
; SSE2-NEXT:  pxor      %xmm4, %xmm4
; SSE2-NEXT:  pcmpgtw   %xmm1, %xmm4
; SSE2-NEXT:  movdqa    %xmm4, %xmm5
; SSE2-NEXT:  pandn     %xmm0, %xmm5
; SSE2-NEXT:  psraw     $2, %xmm0
; SSE2-NEXT:  pand      %xmm4, %xmm0
; SSE2-NEXT:  por       %xmm5, %xmm0
; SSE2-NEXT:  paddw     %xmm1, %xmm1
; SSE2-NEXT:  pcmpgtw   %xmm1, %xmm3
; SSE2-NEXT:  movdqa    %xmm3, %xmm1
; SSE2-NEXT:  pandn     %xmm0, %xmm1
; SSE2-NEXT:  psraw     $1, %xmm0
; SSE2-NEXT:  pand      %xmm3, %xmm0
; SSE2-NEXT:  por       %xmm1, %xmm0
; SSE2-NEXT:  psrlw     $8, %xmm0
; SSE2-NEXT:  packuswb  %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa    %xmm0, %xmm2
; SSE41-NEXT: psllw     $5, %xmm1
; SSE41-NEXT: punpckhbw {{.*#}} xmm0 = xmm0[8],xmm1[8],xmm0[9],xmm1[9],xmm0[10],xmm1[10],xmm0[11],xmm1[11],xmm0[12],xmm1[12],xmm0[13],xmm1[13],xmm0[14],xmm1[14],xmm0[15],xmm1[15]
; SSE41-NEXT: punpckhbw {{.*#}} xmm3 = xmm3[8],xmm2[8],xmm3[9],xmm2[9],xmm3[10],xmm2[10],xmm3[11],xmm2[11],xmm3[12],xmm2[12],xmm3[13],xmm2[13],xmm3[14],xmm2[14],xmm3[15],xmm2[15]
; SSE41-NEXT: movdqa    %xmm3, %xmm4
; SSE41-NEXT: psraw     $4, %xmm4
; SSE41-NEXT: pblendvb  %xmm4, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm4
; SSE41-NEXT: psraw     $2, %xmm4
; SSE41-NEXT: paddw     %xmm0, %xmm0
; SSE41-NEXT: pblendvb  %xmm4, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm4
; SSE41-NEXT: psraw     $1, %xmm4
; SSE41-NEXT: paddw     %xmm0, %xmm0
; SSE41-NEXT: pblendvb  %xmm4, %xmm3
; SSE41-NEXT: psrlw     $8, %xmm3
; SSE41-NEXT: punpcklbw {{.*#}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; SSE41-NEXT: punpcklbw {{.*#}} xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
; SSE41-NEXT: movdqa    %xmm1, %xmm2
; SSE41-NEXT: psraw     $4, %xmm2
; SSE41-NEXT: pblendvb  %xmm2, %xmm1
; SSE41-NEXT: movdqa    %xmm1, %xmm2
; SSE41-NEXT: psraw     $2, %xmm2
; SSE41-NEXT: paddw     %xmm0, %xmm0
; SSE41-NEXT: pblendvb  %xmm2, %xmm1
; SSE41-NEXT: movdqa    %xmm1, %xmm2
; SSE41-NEXT: psraw     $1, %xmm2
; SSE41-NEXT: paddw     %xmm0, %xmm0
; SSE41-NEXT: pblendvb  %xmm2, %xmm1
; SSE41-NEXT: psrlw     $8, %xmm1
; SSE41-NEXT: packuswb  %xmm3, %xmm1
; SSE41-NEXT: movdqa    %xmm1, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw     $5, %xmm1, %xmm1
; AVX-NEXT:   vpunpckhbw {{.*#}} xmm2 = xmm0[8],xmm1[8],xmm0[9],xmm1[9],xmm0[10],xmm1[10],xmm0[11],xmm1[11],xmm0[12],xmm1[12],xmm0[13],xmm1[13],xmm0[14],xmm1[14],xmm0[15],xmm1[15]
; AVX-NEXT:   vpunpckhbw {{.*#}} xmm3 = xmm0[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
; AVX-NEXT:   vpsraw     $4, %xmm3, %xmm4
; AVX-NEXT:   vpblendvb  %xmm2, %xmm4, %xmm3, %xmm3
; AVX-NEXT:   vpsraw     $2, %xmm3, %xmm4
; AVX-NEXT:   vpaddw     %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb  %xmm2, %xmm4, %xmm3, %xmm3
; AVX-NEXT:   vpsraw     $1, %xmm3, %xmm4
; AVX-NEXT:   vpaddw     %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb  %xmm2, %xmm4, %xmm3, %xmm2
; AVX-NEXT:   vpsrlw     $8, %xmm2, %xmm2
; AVX-NEXT:   vpunpcklbw {{.*#}} xmm1 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; AVX-NEXT:   vpunpcklbw {{.*#}} xmm0 = xmm0[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
; AVX-NEXT:   vpsraw     $4, %xmm0, %xmm3
; AVX-NEXT:   vpblendvb  %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsraw     $2, %xmm0, %xmm3
; AVX-NEXT:   vpaddw     %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb  %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsraw     $1, %xmm0, %xmm3
; AVX-NEXT:   vpaddw     %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb  %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw     $8, %xmm0, %xmm0
; AVX-NEXT:   vpackuswb  %xmm2, %xmm0, %xmm0
; AVX-NEXT:   retq
  %ashr = ashr <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %ashr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @lshr_8i16(<8 x i16> %r, <8 x i16> %a) nounwind readnone ssp {
entry:
; ALL-NOT: shrl
;
; SSE2:       psllw   $12, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psrlw   $8, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psrlw   $4, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  psraw   $15, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm3
; SSE2-NEXT:  pandn   %xmm0, %xmm3
; SSE2-NEXT:  psrlw   $2, %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm3, %xmm0
; SSE2-NEXT:  paddw   %xmm1, %xmm1
; SSE2-NEXT:  psraw   $15, %xmm1
; SSE2-NEXT:  movdqa  %xmm1, %xmm2
; SSE2-NEXT:  pandn   %xmm0, %xmm2
; SSE2-NEXT:  psrlw   $1, %xmm0
; SSE2-NEXT:  pand    %xmm1, %xmm0
; SSE2-NEXT:  por     %xmm2, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa    %xmm0, %xmm2
; SSE41-NEXT: movdqa    %xmm1, %xmm0
; SSE41-NEXT: psllw     $12, %xmm0
; SSE41-NEXT: psllw     $4, %xmm1
; SSE41-NEXT: por       %xmm0, %xmm1
; SSE41-NEXT: movdqa    %xmm1, %xmm3
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm2, %xmm4
; SSE41-NEXT: psrlw     $8, %xmm4
; SSE41-NEXT: movdqa    %xmm1, %xmm0
; SSE41-NEXT: pblendvb  %xmm4, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psrlw     $4, %xmm1
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psrlw     $2, %xmm1
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm1
; SSE41-NEXT: psrlw     $1, %xmm1
; SSE41-NEXT: paddw     %xmm3, %xmm3
; SSE41-NEXT: movdqa    %xmm3, %xmm0
; SSE41-NEXT: pblendvb  %xmm1, %xmm2
; SSE41-NEXT: movdqa    %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw    $12, %xmm1, %xmm2
; AVX-NEXT:   vpsllw    $4, %xmm1, %xmm1
; AVX-NEXT:   vpor      %xmm2, %xmm1, %xmm1
; AVX-NEXT:   vpaddw    %xmm1, %xmm1, %xmm2
; AVX-NEXT:   vpsrlw    $8, %xmm0, %xmm3
; AVX-NEXT:   vpblendvb %xmm1, %xmm3, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw    $4, %xmm0, %xmm1
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw    $2, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw    $1, %xmm0, %xmm1
; AVX-NEXT:   vpaddw    %xmm2, %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm2, %xmm1, %xmm0, %xmm0
; AVX-NEXT:   retq
  %lshr = lshr <8 x i16> %r, %a
  %tmp2 = bitcast <8 x i16> %lshr to <2 x i64>
  ret <2 x i64> %tmp2
}

define <2 x i64> @lshr_16i8(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; ALL-NOT: shrb
;
; SSE2:       psllw   $5, %xmm1
; SSE2-NEXT:  pxor    %xmm2, %xmm2
; SSE2-NEXT:  pxor    %xmm3, %xmm3
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm3
; SSE2-NEXT:  movdqa  %xmm3, %xmm4
; SSE2-NEXT:  pandn   %xmm0, %xmm4
; SSE2-NEXT:  psrlw   $4, %xmm0
; SSE2-NEXT:  pand    {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand    %xmm3, %xmm0
; SSE2-NEXT:  por     %xmm4, %xmm0
; SSE2-NEXT:  paddb   %xmm1, %xmm1
; SSE2-NEXT:  pxor    %xmm3, %xmm3
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm3
; SSE2-NEXT:  movdqa  %xmm3, %xmm4
; SSE2-NEXT:  pandn   %xmm0, %xmm4
; SSE2-NEXT:  psrlw   $2, %xmm0
; SSE2-NEXT:  pand    {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand    %xmm3, %xmm0
; SSE2-NEXT:  por     %xmm4, %xmm0
; SSE2-NEXT:  paddb   %xmm1, %xmm1
; SSE2-NEXT:  pcmpgtb %xmm1, %xmm2
; SSE2-NEXT:  movdqa  %xmm2, %xmm1
; SSE2-NEXT:  pandn   %xmm0, %xmm1
; SSE2-NEXT:  psrlw   $1, %xmm0
; SSE2-NEXT:  pand    {{.*}}(%rip), %xmm0
; SSE2-NEXT:  pand    %xmm2, %xmm0
; SSE2-NEXT:  por     %xmm1, %xmm0
; SSE2-NEXT:  retq
;
; SSE41:      movdqa   %xmm0, %xmm2
; SSE41-NEXT: psllw    $5, %xmm1
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: psrlw    $4, %xmm3
; SSE41-NEXT: pand     {{.*}}(%rip), %xmm3
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: psrlw    $2, %xmm3
; SSE41-NEXT: pand     {{.*}}(%rip), %xmm3
; SSE41-NEXT: paddb    %xmm1, %xmm1
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm3
; SSE41-NEXT: psrlw    $1, %xmm3
; SSE41-NEXT: pand     {{.*}}(%rip), %xmm3
; SSE41-NEXT: paddb    %xmm1, %xmm1
; SSE41-NEXT: movdqa   %xmm1, %xmm0
; SSE41-NEXT: pblendvb %xmm3, %xmm2
; SSE41-NEXT: movdqa   %xmm2, %xmm0
; SSE41-NEXT: retq
;
; AVX:        vpsllw    $5, %xmm1, %xmm1
; AVX-NEXT:   vpsrlw    $4, %xmm0, %xmm2
; AVX-NEXT:   vpand     {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw    $2, %xmm0, %xmm2
; AVX-NEXT:   vpand     {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:   vpaddb    %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   vpsrlw    $1, %xmm0, %xmm2
; AVX-NEXT:   vpand     {{.*}}(%rip), %xmm2, %xmm2
; AVX-NEXT:   vpaddb    %xmm1, %xmm1, %xmm1
; AVX-NEXT:   vpblendvb %xmm1, %xmm2, %xmm0, %xmm0
; AVX-NEXT:   retq
  %lshr = lshr <16 x i8> %r, %a
  %tmp2 = bitcast <16 x i8> %lshr to <2 x i64>
  ret <2 x i64> %tmp2
}
