; RUN: llc < %s -mcpu=x86-64 -mattr=+sse2 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE2
; RUN: llc < %s -mcpu=x86-64 -mattr=+ssse3 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSSE3
; RUN: llc < %s -mcpu=x86-64 -mattr=+sse4.1 | FileCheck %s --check-prefix=ALL --check-prefix=SSE --check-prefix=SSE41
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mcpu=x86-64 -mattr=+avx2 | FileCheck %s --check-prefix=ALL --check-prefix=AVX --check-prefix=AVX2
;
; Verify that the DAG combiner correctly folds bitwise operations across
; shuffles, nested shuffles with undef, pairs of nested shuffles, and other
; basic and always-safe patterns. Also test that the DAG combiner will combine
; target-specific shuffle instructions where reasonable.

target triple = "x86_64-unknown-unknown"

declare <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32>, i8)
declare <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16>, i8)
declare <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16>, i8)

define <4 x i32> @combine_pshufd1(<4 x i32> %a) {
; ALL-LABEL: combine_pshufd1:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27)
  %c = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %b, i8 27)
  ret <4 x i32> %c
}

define <4 x i32> @combine_pshufd2(<4 x i32> %a) {
; ALL-LABEL: combine_pshufd2:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27)
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b.cast, i8 -28)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 27)
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd3(<4 x i32> %a) {
; ALL-LABEL: combine_pshufd3:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 27)
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b.cast, i8 -28)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 27)
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd4(<4 x i32> %a) {
; SSE-LABEL: combine_pshufd4:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,7,6,5,4]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_pshufd4:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 -31)
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b.cast, i8 27)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 -31)
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd5(<4 x i32> %a) {
; SSE-LABEL: combine_pshufd5:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[3,2,1,0,4,5,6,7]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_pshufd5:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,2,1,0,4,5,6,7]
; AVX-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 -76)
  %b.cast = bitcast <4 x i32> %b to <8 x i16>
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b.cast, i8 27)
  %c.cast = bitcast <8 x i16> %c to <4 x i32>
  %d = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %c.cast, i8 -76)
  ret <4 x i32> %d
}

define <4 x i32> @combine_pshufd6(<4 x i32> %a) {
; SSE-LABEL: combine_pshufd6:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_pshufd6:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,0,0,0]
; AVX-NEXT:    retq
entry:
  %b = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %a, i8 0)
  %c = call <4 x i32> @llvm.x86.sse2.pshuf.d(<4 x i32> %b, i8 8)
  ret <4 x i32> %c
}

define <8 x i16> @combine_pshuflw1(<8 x i16> %a) {
; ALL-LABEL: combine_pshuflw1:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    retq
entry:
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b, i8 27)
  ret <8 x i16> %c
}

define <8 x i16> @combine_pshuflw2(<8 x i16> %a) {
; ALL-LABEL: combine_pshuflw2:
; ALL:       # BB#0: # %entry
; ALL-NEXT:    retq
entry:
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b, i8 -28)
  %d = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %c, i8 27)
  ret <8 x i16> %d
}

define <8 x i16> @combine_pshuflw3(<8 x i16> %a) {
; SSE-LABEL: combine_pshuflw3:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    pshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,7,6,5,4]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_pshuflw3:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpshufhw {{.*#+}} xmm0 = xmm0[0,1,2,3,7,6,5,4]
; AVX-NEXT:    retq
entry:
  %b = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %b, i8 27)
  %d = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %c, i8 27)
  ret <8 x i16> %d
}

define <8 x i16> @combine_pshufhw1(<8 x i16> %a) {
; SSE-LABEL: combine_pshufhw1:
; SSE:       # BB#0: # %entry
; SSE-NEXT:    pshuflw {{.*#+}} xmm0 = xmm0[3,2,1,0,4,5,6,7]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_pshufhw1:
; AVX:       # BB#0: # %entry
; AVX-NEXT:    vpshuflw {{.*#+}} xmm0 = xmm0[3,2,1,0,4,5,6,7]
; AVX-NEXT:    retq
entry:
  %b = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %a, i8 27)
  %c = call <8 x i16> @llvm.x86.sse2.pshufl.w(<8 x i16> %b, i8 27)
  %d = call <8 x i16> @llvm.x86.sse2.pshufh.w(<8 x i16> %c, i8 27)
  ret <8 x i16> %d
}
