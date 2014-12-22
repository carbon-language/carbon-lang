; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 | FileCheck -check-prefix=AVX2 %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=-popcnt | FileCheck -check-prefix=AVX1-NOPOPCNT %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=-popcnt | FileCheck -check-prefix=AVX2-NOPOPCNT %s

; Vector version of:
; v = v - ((v >> 1) & 0x55555555)
; v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
; v = (v + (v >> 4) & 0xF0F0F0F)
; v = v + (v >> 8)
; v = v + (v >> 16)
; v = v + (v >> 32) ; i64 only

define <8 x i32> @test0(<8 x i32> %x) {
; AVX2-LABEL: @test0
entry:
; AVX2:  vpsrld  $1, %ymm
; AVX2-NEXT:  vpbroadcastd
; AVX2-NEXT:  vpand
; AVX2-NEXT:  vpsubd
; AVX2-NEXT:  vpbroadcastd
; AVX2-NEXT:  vpand
; AVX2-NEXT:  vpsrld  $2
; AVX2-NEXT:  vpand
; AVX2-NEXT:  vpaddd
; AVX2-NEXT:  vpsrld  $4
; AVX2-NEXT:  vpaddd
; AVX2-NEXT:  vpbroadcastd
; AVX2-NEXT:	vpand
; AVX2-NEXT:	vpsrld	$8
; AVX2-NEXT:	vpaddd
; AVX2-NEXT:	vpsrld	$16
; AVX2-NEXT:	vpaddd
; AVX2-NEXT:	vpbroadcastd
; AVX2-NEXT:	vpand
  %y = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %x)
  ret <8 x i32> %y
}

define <4 x i64> @test1(<4 x i64> %x) {
; AVX2-NOPOPCNT-LABEL: @test1
entry:
;	AVX2-NOPOPCNT: vpsrlq	$1, %ymm
;	AVX2-NOPOPCNT-NEXT: vpbroadcastq
;	AVX2-NOPOPCNT-NEXT: vpand
;	AVX2-NOPOPCNT-NEXT: vpsubq
;	AVX2-NOPOPCNT-NEXT: vpbroadcastq
;	AVX2-NOPOPCNT-NEXT: vpand
;	AVX2-NOPOPCNT-NEXT: vpsrlq	$2
;	AVX2-NOPOPCNT-NEXT: vpand
;	AVX2-NOPOPCNT-NEXT: vpaddq
;	AVX2-NOPOPCNT-NEXT: vpsrlq	$4
;	AVX2-NOPOPCNT-NEXT: vpaddq
;	AVX2-NOPOPCNT-NEXT: vpbroadcastq
;	AVX2-NOPOPCNT-NEXT: vpand
;	AVX2-NOPOPCNT-NEXT: vpsrlq	$8
;	AVX2-NOPOPCNT-NEXT: vpaddq
;	AVX2-NOPOPCNT-NEXT: vpsrlq	$16
;	AVX2-NOPOPCNT-NEXT: vpaddq
;	AVX2-NOPOPCNT-NEXT: vpsrlq	$32
;	AVX2-NOPOPCNT-NEXT: vpaddq
;	AVX2-NOPOPCNT-NEXT: vpbroadcastq
;	AVX2-NOPOPCNT-NEXT: vpand
  %y = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %x)
  ret <4 x i64> %y
}

define <4 x i32> @test2(<4 x i32> %x) {
; AVX2-NOPOPCNT-LABEL: @test2
; AVX1-NOPOPCNT-LABEL: @test2
entry:
; AVX2-NOPOPCNT:	vpsrld	$1, %xmm
; AVX2-NOPOPCNT-NEXT:	vpbroadcastd
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsubd
; AVX2-NOPOPCNT-NEXT:	vpbroadcastd
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsrld	$2
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpaddd
; AVX2-NOPOPCNT-NEXT:	vpsrld	$4
; AVX2-NOPOPCNT-NEXT:	vpaddd
; AVX2-NOPOPCNT-NEXT:	vpbroadcastd
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsrld	$8
; AVX2-NOPOPCNT-NEXT:	vpaddd
; AVX2-NOPOPCNT-NEXT:	vpsrld	$16
; AVX2-NOPOPCNT-NEXT:	vpaddd
; AVX2-NOPOPCNT-NEXT:	vpbroadcastd
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT:	vpsrld	$1, %xmm
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsubd
; AVX1-NOPOPCNT-NEXT:	vmovdqa
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsrld	$2
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpaddd
; AVX1-NOPOPCNT-NEXT:	vpsrld	$4
; AVX1-NOPOPCNT-NEXT:	vpaddd
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsrld	$8
; AVX1-NOPOPCNT-NEXT:	vpaddd
; AVX1-NOPOPCNT-NEXT:	vpsrld	$16
; AVX1-NOPOPCNT-NEXT:	vpaddd
; AVX1-NOPOPCNT-NEXT:	vpand
  %y = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %x)
  ret <4 x i32> %y
}

define <2 x i64> @test3(<2 x i64> %x) {
; AVX2-NOPOPCNT-LABEL: @test3
; AVX1-NOPOPCNT-LABEL: @test3
entry:
; AVX2-NOPOPCNT:	vpsrlq	$1, %xmm
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsubq
; AVX2-NOPOPCNT-NEXT:	vmovdqa
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsrlq	$2
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpaddq
; AVX2-NOPOPCNT-NEXT:	vpsrlq	$4
; AVX2-NOPOPCNT-NEXT:	vpaddq
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX2-NOPOPCNT-NEXT:	vpsrlq	$8
; AVX2-NOPOPCNT-NEXT:	vpaddq
; AVX2-NOPOPCNT-NEXT:	vpsrlq	$16
; AVX2-NOPOPCNT-NEXT:	vpaddq
; AVX2-NOPOPCNT-NEXT:	vpsrlq	$32
; AVX2-NOPOPCNT-NEXT:	vpaddq
; AVX2-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT:	vpsrlq	$1, %xmm
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsubq
; AVX1-NOPOPCNT-NEXT:	vmovdqa
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsrlq	$2
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpaddq
; AVX1-NOPOPCNT-NEXT:	vpsrlq	$4
; AVX1-NOPOPCNT-NEXT:	vpaddq
; AVX1-NOPOPCNT-NEXT:	vpand
; AVX1-NOPOPCNT-NEXT:	vpsrlq	$8
; AVX1-NOPOPCNT-NEXT:	vpaddq
; AVX1-NOPOPCNT-NEXT:	vpsrlq	$16
; AVX1-NOPOPCNT-NEXT:	vpaddq
; AVX1-NOPOPCNT-NEXT:	vpsrlq	$32
; AVX1-NOPOPCNT-NEXT:	vpaddq
; AVX1-NOPOPCNT-NEXT:	vpand
  %y = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %x)
  ret <2 x i64> %y
}

declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>)
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>)

declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>)
declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>)

