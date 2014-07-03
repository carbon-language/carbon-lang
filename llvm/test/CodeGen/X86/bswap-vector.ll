; RUN: llc < %s -mcpu=x86-64 | FileCheck %s -check-prefix=CHECK-NOSSSE3
; RUN: llc < %s -mcpu=core2 | FileCheck %s -check-prefix=CHECK-SSSE3
; RUN: llc < %s -mcpu=core-avx2 | FileCheck %s -check-prefix=CHECK-AVX2
; RUN: llc < %s -mcpu=core-avx2 -x86-experimental-vector-widening-legalization | FileCheck %s -check-prefix=CHECK-WIDE-AVX2
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <8 x i16> @llvm.bswap.v8i16(<8 x i16>)
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>)
declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

define <8 x i16> @test1(<8 x i16> %v) #0 {
entry:
  %r = call <8 x i16> @llvm.bswap.v8i16(<8 x i16> %v)
  ret <8 x i16> %r

; CHECK-NOSSSE3-LABEL: @test1
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: rolw
; CHECK-NOSSSE3: retq

; CHECK-SSSE3-LABEL: @test1
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test1
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test1
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

define <4 x i32> @test2(<4 x i32> %v) #0 {
entry:
  %r = call <4 x i32> @llvm.bswap.v4i32(<4 x i32> %v)
  ret <4 x i32> %r

; CHECK-NOSSSE3-LABEL: @test2
; CHECK-NOSSSE3: bswapl
; CHECK-NOSSSE3: bswapl
; CHECK-NOSSSE3: bswapl
; CHECK-NOSSSE3: bswapl
; CHECK-NOSSSE3: retq

; CHECK-SSSE3-LABEL: @test2
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test2
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test2
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

define <2 x i64> @test3(<2 x i64> %v) #0 {
entry:
  %r = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %v)
  ret <2 x i64> %r

; CHECK-NOSSSE3-LABEL: @test3
; CHECK-NOSSSE3: bswapq
; CHECK-NOSSSE3: bswapq
; CHECK-NOSSSE3: retq

; CHECK-SSSE3-LABEL: @test3
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test3
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test3
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

declare <16 x i16> @llvm.bswap.v16i16(<16 x i16>)
declare <8 x i32> @llvm.bswap.v8i32(<8 x i32>)
declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>)

define <16 x i16> @test4(<16 x i16> %v) #0 {
entry:
  %r = call <16 x i16> @llvm.bswap.v16i16(<16 x i16> %v)
  ret <16 x i16> %r

; CHECK-SSSE3-LABEL: @test4
; CHECK-SSSE3: pshufb
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test4
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test4
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

define <8 x i32> @test5(<8 x i32> %v) #0 {
entry:
  %r = call <8 x i32> @llvm.bswap.v8i32(<8 x i32> %v)
  ret <8 x i32> %r

; CHECK-SSSE3-LABEL: @test5
; CHECK-SSSE3: pshufb
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test5
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test5
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

define <4 x i64> @test6(<4 x i64> %v) #0 {
entry:
  %r = call <4 x i64> @llvm.bswap.v4i64(<4 x i64> %v)
  ret <4 x i64> %r

; CHECK-SSSE3-LABEL: @test6
; CHECK-SSSE3: pshufb
; CHECK-SSSE3: pshufb
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test6
; CHECK-AVX2: vpshufb
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test6
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>)

define <4 x i16> @test7(<4 x i16> %v) #0 {
entry:
  %r = call <4 x i16> @llvm.bswap.v4i16(<4 x i16> %v)
  ret <4 x i16> %r

; CHECK-SSSE3-LABEL: @test7
; CHECK-SSSE3: pshufb
; CHECK-SSSE3: psrld $16
; CHECK-SSSE3-NEXT: retq

; CHECK-AVX2-LABEL: @test7
; CHECK-AVX2: vpshufb
; CHECK-AVX2: vpsrld $16
; CHECK-AVX2-NEXT: retq

; CHECK-WIDE-AVX2-LABEL: @test7
; CHECK-WIDE-AVX2: vpshufb
; CHECK-WIDE-AVX2-NEXT: retq
}

attributes #0 = { nounwind uwtable }

