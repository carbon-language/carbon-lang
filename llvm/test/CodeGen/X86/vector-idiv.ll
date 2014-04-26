; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE
; RUN: llc -march=x86-64 -mcpu=core-avx2 < %s | FileCheck %s -check-prefix=AVX

define <4 x i32> @test1(<4 x i32> %a) {
  %div = udiv <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %div

; SSE-LABEL: test1:
; SSE: pmuludq
; SSE: pshufd	$57
; SSE: pmuludq
; SSE: shufps	$-35
; SSE: psubd
; SSE: psrld $1
; SSE: padd
; SSE: psrld $2

; AVX-LABEL: test1:
; AVX: vpmuludq
; AVX: vpshufd	$57
; AVX: vpmuludq
; AVX: vshufps	$-35
; AVX: vpsubd
; AVX: vpsrld $1
; AVX: vpadd
; AVX: vpsrld $2
}

define <8 x i32> @test2(<8 x i32> %a) {
  %div = udiv <8 x i32> %a, <i32 7, i32 7, i32 7, i32 7,i32 7, i32 7, i32 7, i32 7>
  ret <8 x i32> %div

; AVX-LABEL: test2:
; AVX: vpermd
; AVX: vpmuludq
; AVX: vshufps	$-35
; AVX: vpmuludq
; AVX: vshufps	$-35
; AVX: vpsubd
; AVX: vpsrld $1
; AVX: vpadd
; AVX: vpsrld $2
}

define <8 x i16> @test3(<8 x i16> %a) {
  %div = udiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; SSE-LABEL: test3:
; SSE: pmulhuw
; SSE: psubw
; SSE: psrlw $1
; SSE: paddw
; SSE: psrlw $2

; AVX-LABEL: test3:
; AVX: vpmulhuw
; AVX: vpsubw
; AVX: vpsrlw $1
; AVX: vpaddw
; AVX: vpsrlw $2
}

define <16 x i16> @test4(<16 x i16> %a) {
  %div = udiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; AVX-LABEL: test4:
; AVX: vpmulhuw
; AVX: vpsubw
; AVX: vpsrlw $1
; AVX: vpaddw
; AVX: vpsrlw $2
; AVX-NOT: vpmulhuw
}

define <8 x i16> @test5(<8 x i16> %a) {
  %div = sdiv <8 x i16> %a, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  ret <8 x i16> %div

; SSE-LABEL: test5:
; SSE: pmulhw
; SSE: psrlw $15
; SSE: psraw $1
; SSE: paddw

; AVX-LABEL: test5:
; AVX: vpmulhw
; AVX: vpsrlw $15
; AVX: vpsraw $1
; AVX: vpaddw
}

define <16 x i16> @test6(<16 x i16> %a) {
  %div = sdiv <16 x i16> %a, <i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7,i16 7, i16 7, i16 7, i16 7>
  ret <16 x i16> %div

; AVX-LABEL: test6:
; AVX: vpmulhw
; AVX: vpsrlw $15
; AVX: vpsraw $1
; AVX: vpaddw
; AVX-NOT: vpmulhw
}

; TODO: sdiv -> pmuldq
