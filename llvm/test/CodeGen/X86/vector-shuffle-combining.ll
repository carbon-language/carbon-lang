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

define <4 x i32> @combine_bitwise_ops_test1(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test1:
; SSE:       # BB#0:
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test1:
; AVX:       # BB#0:
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test2(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test2:
; SSE:       # BB#0:
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test2:
; AVX:       # BB#0:
; AVX-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test3(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test3:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test3:
; AVX:       # BB#0:
; AVX-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 1, i32 3>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}

define <4 x i32> @combine_bitwise_ops_test4(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test4:
; SSE:       # BB#0:
; SSE-NEXT:    pand %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test4:
; AVX:       # BB#0:
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test5(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test5:
; SSE:       # BB#0:
; SSE-NEXT:    por %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test5:
; AVX:       # BB#0:
; AVX-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test6(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test6:
; SSE:       # BB#0:
; SSE-NEXT:    pxor %xmm1, %xmm0
; SSE-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test6:
; AVX:       # BB#0:
; AVX-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 4, i32 6, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}


; Verify that DAGCombiner moves the shuffle after the xor/and/or even if shuffles
; are not performing a swizzle operations.

define <4 x i32> @combine_bitwise_ops_test1b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test1b:
; SSE2:       # BB#0:
; SSE2-NEXT:    andps %xmm1, %xmm0
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test1b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    andps %xmm1, %xmm0
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test1b:
; SSE41:       # BB#0:
; SSE41-NEXT:    andps %xmm1, %xmm0
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test1b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vandps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test1b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test2b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test2b:
; SSE2:       # BB#0:
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test2b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    orps %xmm1, %xmm0
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test2b:
; SSE41:       # BB#0:
; SSE41-NEXT:    orps %xmm1, %xmm0
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test2b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test2b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm2[1],xmm0[2],xmm2[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test3b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test3b:
; SSE2:       # BB#0:
; SSE2-NEXT:    xorps %xmm1, %xmm0
; SSE2-NEXT:    xorps %xmm1, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test3b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    xorps %xmm1, %xmm0
; SSSE3-NEXT:    xorps %xmm1, %xmm1
; SSSE3-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm0[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test3b:
; SSE41:       # BB#0:
; SSE41-NEXT:    xorps %xmm1, %xmm0
; SSE41-NEXT:    xorps %xmm1, %xmm1
; SSE41-NEXT:    blendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test3b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test3b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm0[0],xmm1[1],xmm0[2],xmm1[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}

define <4 x i32> @combine_bitwise_ops_test4b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test4b:
; SSE2:       # BB#0:
; SSE2-NEXT:    andps %xmm1, %xmm0
; SSE2-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test4b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    andps %xmm1, %xmm0
; SSSE3-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test4b:
; SSE41:       # BB#0:
; SSE41-NEXT:    andps %xmm1, %xmm0
; SSE41-NEXT:    blendps {{.*#+}} xmm2 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; SSE41-NEXT:    movaps %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test4b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vandps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test4b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test5b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test5b:
; SSE2:       # BB#0:
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test5b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    orps %xmm1, %xmm0
; SSSE3-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm2[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test5b:
; SSE41:       # BB#0:
; SSE41-NEXT:    orps %xmm1, %xmm0
; SSE41-NEXT:    blendps {{.*#+}} xmm2 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; SSE41-NEXT:    movaps %xmm2, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test5b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test5b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm2[0],xmm0[1],xmm2[2],xmm0[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test6b(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE2-LABEL: combine_bitwise_ops_test6b:
; SSE2:       # BB#0:
; SSE2-NEXT:    xorps %xmm1, %xmm0
; SSE2-NEXT:    xorps %xmm1, %xmm1
; SSE2-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,2],xmm0[1,3]
; SSE2-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,2,1,3]
; SSE2-NEXT:    retq
;
; SSSE3-LABEL: combine_bitwise_ops_test6b:
; SSSE3:       # BB#0:
; SSSE3-NEXT:    xorps %xmm1, %xmm0
; SSSE3-NEXT:    xorps %xmm1, %xmm1
; SSSE3-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,2],xmm0[1,3]
; SSSE3-NEXT:    pshufd {{.*#+}} xmm0 = xmm1[0,2,1,3]
; SSSE3-NEXT:    retq
;
; SSE41-LABEL: combine_bitwise_ops_test6b:
; SSE41:       # BB#0:
; SSE41-NEXT:    xorps %xmm1, %xmm0
; SSE41-NEXT:    xorps %xmm1, %xmm1
; SSE41-NEXT:    blendps {{.*#+}} xmm1 = xmm1[0],xmm0[1],xmm1[2],xmm0[3]
; SSE41-NEXT:    movaps %xmm1, %xmm0
; SSE41-NEXT:    retq
;
; AVX1-LABEL: combine_bitwise_ops_test6b:
; AVX1:       # BB#0:
; AVX1-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; AVX1-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vblendps {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: combine_bitwise_ops_test6b:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX2-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vpblendd {{.*#+}} xmm0 = xmm1[0],xmm0[1],xmm1[2],xmm0[3]
; AVX2-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 5, i32 2, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}

define <4 x i32> @combine_bitwise_ops_test1c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test1c:
; SSE:       # BB#0:
; SSE-NEXT:    andps %xmm1, %xmm0
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test1c:
; AVX:       # BB#0:
; AVX-NEXT:    vandps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test2c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test2c:
; SSE:       # BB#0:
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test2c:
; AVX:       # BB#0:
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm0[0,2],xmm2[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test3c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test3c:
; SSE:       # BB#0:
; SSE-NEXT:    xorps %xmm1, %xmm0
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test3c:
; AVX:       # BB#0:
; AVX-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %a, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %b, <4 x i32> %c, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}

define <4 x i32> @combine_bitwise_ops_test4c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test4c:
; SSE:       # BB#0:
; SSE-NEXT:    andps %xmm1, %xmm0
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test4c:
; AVX:       # BB#0:
; AVX-NEXT:    vandps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm2[0,2],xmm0[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %and = and <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %and
}

define <4 x i32> @combine_bitwise_ops_test5c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test5c:
; SSE:       # BB#0:
; SSE-NEXT:    orps %xmm1, %xmm0
; SSE-NEXT:    shufps {{.*#+}} xmm2 = xmm2[0,2],xmm0[1,3]
; SSE-NEXT:    movaps %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test5c:
; AVX:       # BB#0:
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm2[0,2],xmm0[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %or = or <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %or
}

define <4 x i32> @combine_bitwise_ops_test6c(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
; SSE-LABEL: combine_bitwise_ops_test6c:
; SSE:       # BB#0:
; SSE-NEXT:    xorps %xmm1, %xmm0
; SSE-NEXT:    xorps %xmm1, %xmm1
; SSE-NEXT:    shufps {{.*#+}} xmm1 = xmm1[0,2],xmm0[1,3]
; SSE-NEXT:    movaps %xmm1, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: combine_bitwise_ops_test6c:
; AVX:       # BB#0:
; AVX-NEXT:    vxorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vshufps {{.*#+}} xmm0 = xmm1[0,2],xmm0[1,3]
; AVX-NEXT:    retq
  %shuf1 = shufflevector <4 x i32> %c, <4 x i32> %a, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %shuf2 = shufflevector <4 x i32> %c, <4 x i32> %b, <4 x i32><i32 0, i32 2, i32 5, i32 7>
  %xor = xor <4 x i32> %shuf1, %shuf2
  ret <4 x i32> %xor
}
