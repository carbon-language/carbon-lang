; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse2 | FileCheck %s -check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse41 | FileCheck %s -check-prefix=SSE41
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=avx | FileCheck %s -check-prefix=AVX

define <16 x i8> @v16i8_icmp_uge(<16 x i8> %a, <16 x i8> %b) nounwind readnone ssp uwtable {
  %1 = icmp uge <16 x i8> %a, %b
  %2 = sext <16 x i1> %1 to <16 x i8>
  ret <16 x i8> %2
; SSE2-LABEL: v16i8_icmp_uge:
; SSE2: pmaxub  %xmm0, %xmm1
; SSE2: pcmpeqb %xmm1, %xmm0

; SSE41-LABEL: v16i8_icmp_uge:
; SSE41: pmaxub  %xmm0, %xmm1
; SSE41: pcmpeqb %xmm1, %xmm0

; AVX-LABEL: v16i8_icmp_uge:
; AVX: vpmaxub  %xmm1, %xmm0, %xmm1
; AVX: vpcmpeqb %xmm1, %xmm0, %xmm0
}

define <16 x i8> @v16i8_icmp_ule(<16 x i8> %a, <16 x i8> %b) nounwind readnone ssp uwtable {
  %1 = icmp ule <16 x i8> %a, %b
  %2 = sext <16 x i1> %1 to <16 x i8>
  ret <16 x i8> %2
; SSE2-LABEL: v16i8_icmp_ule:
; SSE2: pminub  %xmm0, %xmm1
; SSE2: pcmpeqb %xmm1, %xmm0

; SSE41-LABEL: v16i8_icmp_ule:
; SSE41: pminub  %xmm0, %xmm1
; SSE41: pcmpeqb %xmm1, %xmm0

; AVX-LABEL: v16i8_icmp_ule:
; AVX: vpminub  %xmm1, %xmm0, %xmm1
; AVX: vpcmpeqb %xmm1, %xmm0, %xmm0
}


define <8 x i16> @v8i16_icmp_uge(<8 x i16> %a, <8 x i16> %b) nounwind readnone ssp uwtable {
  %1 = icmp uge <8 x i16> %a, %b
  %2 = sext <8 x i1> %1 to <8 x i16>
  ret <8 x i16> %2
; SSE2-LABEL: v8i16_icmp_uge:
; SSE2: movdqa  {{.*}}(%rip), %xmm2
; SEE2: pxor    %xmm2, %xmm0
; SSE2: pxor    %xmm1, %xmm2
; SSE2: pcmpgtw %xmm0, %xmm2
; SSE2: pcmpeqd %xmm0, %xmm0
; SSE2: pxor    %xmm2, %xmm0

; SSE41-LABEL: v8i16_icmp_uge:
; SSE41: pmaxuw  %xmm0, %xmm1
; SSE41: pcmpeqw %xmm1, %xmm0

; AVX-LABEL: v8i16_icmp_uge:
; AVX: vpmaxuw  %xmm1, %xmm0, %xmm1
; AVX: vpcmpeqw %xmm1, %xmm0, %xmm0
}

define <8 x i16> @v8i16_icmp_ule(<8 x i16> %a, <8 x i16> %b) nounwind readnone ssp uwtable {
  %1 = icmp ule <8 x i16> %a, %b
  %2 = sext <8 x i1> %1 to <8 x i16>
  ret <8 x i16> %2
; SSE2-LABEL: v8i16_icmp_ule:
; SSE2: movdqa  {{.*}}(%rip), %xmm2
; SSE2: pxor    %xmm2, %xmm1
; SSE2: pxor    %xmm2, %xmm0
; SSE2: pcmpgtw %xmm1, %xmm0
; SSE2: pcmpeqd %xmm1, %xmm1
; SSE2: pxor    %xmm0, %xmm1
; SSE2: movdqa  %xmm1, %xmm0

; SSE41-LABEL: v8i16_icmp_ule:
; SSE41: pminuw  %xmm0, %xmm1
; SSE41: pcmpeqw %xmm1, %xmm0

; AVX-LABEL: v8i16_icmp_ule:
; AVX: vpminuw  %xmm1, %xmm0, %xmm1
; AVX: vpcmpeqw %xmm1, %xmm0, %xmm0
}


define <4 x i32> @v4i32_icmp_uge(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp uwtable {
  %1 = icmp uge <4 x i32> %a, %b
  %2 = sext <4 x i1> %1 to <4 x i32>
  ret <4 x i32> %2
; SSE2-LABEL: v4i32_icmp_uge:
; SSE2: movdqa  {{.*}}(%rip), %xmm2
; SSE2: pxor    %xmm2, %xmm0
; SSE2: pxor    %xmm1, %xmm2
; SSE2: pcmpgtd %xmm0, %xmm2
; SSE2: pcmpeqd %xmm0, %xmm0
; SSE2: pxor    %xmm2, %xmm0

; SSE41-LABEL: v4i32_icmp_uge:
; SSE41: pmaxud  %xmm0, %xmm1
; SSE41: pcmpeqd %xmm1, %xmm0

; AVX-LABEL: v4i32_icmp_uge:
; AVX: vpmaxud  %xmm1, %xmm0, %xmm1
; AVX: vpcmpeqd %xmm1, %xmm0, %xmm0
}

define <4 x i32> @v4i32_icmp_ule(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp uwtable {
  %1 = icmp ule <4 x i32> %a, %b
  %2 = sext <4 x i1> %1 to <4 x i32>
  ret <4 x i32> %2
; SSE2-LABEL: v4i32_icmp_ule:
; SSE2: movdqa  {{.*}}(%rip), %xmm2
; SSE2: pxor    %xmm2, %xmm1
; SSE2: pxor    %xmm2, %xmm0
; SSE2: pcmpgtd %xmm1, %xmm0
; SSE2: pcmpeqd %xmm1, %xmm1
; SSE2: pxor    %xmm0, %xmm1
; SSE2: movdqa  %xmm1, %xmm0

; SSE41-LABEL: v4i32_icmp_ule:
; SSE41: pminud  %xmm0, %xmm1
; SSE41: pcmpeqd %xmm1, %xmm0

; AVX-LABEL: v4i32_icmp_ule:
; AVX: pminud  %xmm1, %xmm0, %xmm1
; AVX: pcmpeqd %xmm1, %xmm0, %xmm0
}

