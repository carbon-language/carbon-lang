; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -x86-experimental-vector-shuffle-lowering | FileCheck %s --check-prefix=CHECK-SSE2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

declare <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32>, i8)
declare <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16>, i8)
declare <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16>, i8)

define <4 x i32> @combine_pshufd1(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd1
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27) 
  %c = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %b, i8 27) 
  ret <4 x i32> %c
}

define <4 x i32> @combine_pshufd2(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd2
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27) 
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b.cast, i8 -28)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 27) 
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd3(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd3
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27) 
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b.cast, i8 -28)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 27) 
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd4(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd4
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 -31) 
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b.cast, i8 27)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 -31) 
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd5(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd5
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 -76) 
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b.cast, i8 27)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 -76)
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd6(<4 x i32> %a) {
; CHECK-SSE2-LABEL: @combine_pshufd6
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufd $0
; CHECK-SSE2-NEXT:    retq
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 0)
  %c = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %b, i8 8)
  ret <4 x i32> %c
}

define <8 x i16> @combine_pshuflw1(<8 x i16> %a) {
; CHECK-SSE2-LABEL: @combine_pshuflw1
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    retq
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27) 
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b, i8 27) 
  ret <8 x i16> %c
}

define <8 x i16> @combine_pshuflw2(<8 x i16> %a) {
; CHECK-SSE2-LABEL: @combine_pshuflw2
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    retq
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b, i8 -28) 
  %d = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %c, i8 27) 
  ret <8 x i16> %d
}

define <8 x i16> @combine_pshuflw3(<8 x i16> %a) {
; CHECK-SSE2-LABEL: @combine_pshuflw3
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshufhw {{.*}} # xmm0 = xmm0[0,1,2,3,7,6,5,4]
; CHECK-SSE2-NEXT:    retq
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b, i8 27) 
  %d = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %c, i8 27) 
  ret <8 x i16> %d
}

define <8 x i16> @combine_pshufhw1(<8 x i16> %a) {
; CHECK-SSE2-LABEL: @combine_pshufhw1
; CHECK-SSE2:       # BB#0:
; CHECK-SSE2-NEXT:    pshuflw {{.*}} # xmm0 = xmm0[3,2,1,0,4,5,6,7]
; CHECK-SSE2-NEXT:    retq
  %b = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b, i8 27) 
  %d = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %c, i8 27) 
  ret <8 x i16> %d
}

