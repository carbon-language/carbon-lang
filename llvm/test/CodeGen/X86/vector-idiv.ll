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

; TODO: sdiv -> pmuldq
