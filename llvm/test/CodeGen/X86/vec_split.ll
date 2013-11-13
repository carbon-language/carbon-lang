; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s -check-prefix=SSE4
; RUN: llc -march=x86-64 -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -march=x86-64 -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX2

define <16 x i16> @split16(<16 x i16> %a, <16 x i16> %b, <16 x i8> %__mask) {
; SSE4-LABEL: split16:
; SSE4: pminuw
; SSE4: pminuw
; SSE4: ret
; AVX1-LABEL: split16:
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: ret
; AVX2-LABEL: split16:
; AVX2: vpminuw
; AVX2: ret
  %1 = icmp ult <16 x i16> %a, %b
  %2 = select <16 x i1> %1, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %2
}

define <32 x i16> @split32(<32 x i16> %a, <32 x i16> %b, <32 x i8> %__mask) {
; SSE4-LABEL: split32:
; SSE4: pminuw
; SSE4: pminuw
; SSE4: pminuw
; SSE4: pminuw
; SSE4: ret
; AVX1-LABEL: split32:
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: vpminuw
; AVX1: ret
; AVX2-LABEL: split32:
; AVX2: vpminuw
; AVX2: vpminuw
; AVX2: ret
  %1 = icmp ult <32 x i16> %a, %b
  %2 = select <32 x i1> %1, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %2
}
