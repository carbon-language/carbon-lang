; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2 -check-prefix=AVX2ONLY
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=knl | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2 -check-prefix=AVX512


; Verify that we don't scalarize a packed vector shift left of 16-bit
; signed integers if the amount is a constant build_vector.
; Check that we produce a SSE2 packed integer multiply (pmullw) instead.

define <8 x i16> @test1(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <8 x i16> %shl
}
; CHECK-LABEL: test1
; CHECK: pmullw
; CHECK-NEXT: ret


define <8 x i16> @test2(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 0, i16 undef, i16 0, i16 0, i16 1, i16 undef, i16 -1, i16 1>
  ret <8 x i16> %shl
}
; CHECK-LABEL: test2
; CHECK: pmullw
; CHECK-NEXT: ret


; Verify that a vector shift left of 32-bit signed integers is simply expanded
; into a SSE4.1 pmulld (instead of cvttps2dq + pmulld) if the vector of shift
; counts is a constant build_vector.

define <4 x i32> @test3(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 1, i32 -1, i32 2, i32 -3>
  ret <4 x i32> %shl
}
; CHECK-LABEL: test3
; CHECK-NOT: cvttps2dq
; SSE: pmulld
; AVX2: vpsllvd
; CHECK-NEXT: ret


define <4 x i32> @test4(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 0, i32 0, i32 1, i32 1>
  ret <4 x i32> %shl
}
; CHECK-LABEL: test4
; CHECK-NOT: cvttps2dq
; SSE: pmulld
; AVX2: vpsllvd
; CHECK-NEXT: ret


; If we have AVX/SSE2 but not AVX2, verify that the following shift is split
; into two pmullw instructions. With AVX2, the test case below would produce
; a single vpmullw.

define <16 x i16> @test5(<16 x i16> %a) {
  %shl = shl <16 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <16 x i16> %shl
}
; CHECK-LABEL: test5
; SSE: pmullw
; SSE-NEXT: pmullw
; AVX2: vpmullw
; AVX2-NOT: vpmullw
; CHECK: ret


; If we have AVX/SSE4.1 but not AVX2, verify that the following shift is split
; into two pmulld instructions. With AVX2, the test case below would produce
; a single vpsllvd instead.

define <8 x i32> @test6(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <8 x i32> %shl
}
; CHECK-LABEL: test6
; SSE: pmulld
; SSE-NEXT: pmulld
; AVX2: vpsllvd
; CHECK: ret


; With AVX2 and AVX512, the test case below should produce a sequence of
; two vpmullw instructions. On SSE2 instead, we split the shift in four
; parts and then we convert each part into a pmullw.

define <32 x i16> @test7(<32 x i16> %a) {
  %shl = shl <32 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <32 x i16> %shl
}
; CHECK-LABEL: test7
; SSE: pmullw
; SSE-NEXT: pmullw
; SSE-NEXT: pmullw
; SSE-NEXT: pmullw
; AVX2: vpmullw
; AVX2-NEXT: vpmullw
; CHECK: ret


; Similar to test7; the difference is that with AVX512 support
; we only produce a single vpsllvd/vpsllvq instead of a pair of vpsllvd/vpsllvq.

define <16 x i32> @test8(<16 x i32> %a) {
  %shl = shl <16 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <16 x i32> %shl
}
; CHECK-LABEL: test8
; SSE: pmulld
; SSE-NEXT: pmulld
; SSE-NEXT: pmulld
; SSE-NEXT: pmulld
; AVX2ONLY: vpsllvd
; AVX2ONLY-NEXT: vpsllvd
; AVX512: vpsllvd
; AVX512-NOT: vpsllvd
; CHECK: ret


; The shift from 'test9' gets scalarized if we don't have AVX2/AVX512f support.

define <8 x i64> @test9(<8 x i64> %a) {
  %shl = shl <8 x i64> %a, <i64 1, i64 1, i64 2, i64 3, i64 1, i64 1, i64 2, i64 3>
  ret <8 x i64> %shl
}
; CHECK-LABEL: test9
; AVX2ONLY: vpsllvq
; AVX2ONLY-NEXT: vpsllvq
; AVX512: vpsllvq
; AVX512-NOT: vpsllvq
; CHECK: ret

