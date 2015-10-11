; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2,-sse4.1 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE41
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=bdver4 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=XOP -check-prefix=XOPAVX2


; Verify the cost of vector shift left instructions.

; We always emit a single pmullw in the case of v8i16 vector shifts by
; non-uniform constant.

define <8 x i16> @test1(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <8 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test1':
; CHECK: Found an estimated cost of 1 for instruction:   %shl


define <8 x i16> @test2(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 0, i16 undef, i16 0, i16 0, i16 1, i16 undef, i16 -1, i16 1>
  ret <8 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test2':
; CHECK: Found an estimated cost of 1 for instruction:   %shl


; With SSE4.1, v4i32 shifts can be lowered into a single pmulld instruction.
; Make sure that the estimated cost is always 1 except for the case where
; we only have SSE2 support. With SSE2, we are forced to special lower the
; v4i32 mul as a 2x shuffle, 2x pmuludq, 2x shuffle.

define <4 x i32> @test3(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 1, i32 -1, i32 2, i32 -3>
  ret <4 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test3':
; SSE2: Found an estimated cost of 6 for instruction:   %shl
; SSE41: Found an estimated cost of 1 for instruction:   %shl
; AVX: Found an estimated cost of 1 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


define <4 x i32> @test4(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 0, i32 0, i32 1, i32 1>
  ret <4 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test4':
; SSE2: Found an estimated cost of 6 for instruction:   %shl
; SSE41: Found an estimated cost of 1 for instruction:   %shl
; AVX: Found an estimated cost of 1 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a single
; vpsllvq. Therefore, the expected cost is only 1.
; In all other cases, this shift is scalarized as the target does not support
; vpsllv instructions.

define <2 x i64> @test5(<2 x i64> %a) {
  %shl = shl <2 x i64> %a, <i64 2, i64 3>
  ret <2 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test5':
; SSE2: Found an estimated cost of 4 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


; v16i16 and v8i32 shift left by non-uniform constant are lowered into
; vector multiply instructions.  With AVX (but not AVX2), the vector multiply
; is lowered into a sequence of: 1 extract + 2 vpmullw + 1 insert.
;
; With AVX2, instruction vpmullw works with 256bit quantities and
; therefore there is no need to split the resulting vector multiply into
; a sequence of two multiply.
;
; With SSE2 and SSE4.1, the vector shift cost for 'test6' is twice
; the cost computed in the case of 'test1'. That is because the backend
; simply emits 2 pmullw with no extract/insert.


define <16 x i16> @test6(<16 x i16> %a) {
  %shl = shl <16 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <16 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test6':
; SSE2: Found an estimated cost of 2 for instruction:   %shl
; SSE41: Found an estimated cost of 2 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; With SSE2 and SSE4.1, the vector shift cost for 'test7' is twice
; the cost computed in the case of 'test3'. That is because the multiply
; is type-legalized into two 4i32 vector multiply.

define <8 x i32> @test7(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <8 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test7':
; SSE2: Found an estimated cost of 12 for instruction:   %shl
; SSE41: Found an estimated cost of 2 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a single
; vpsllvq. Therefore, the expected cost is only 1.
; In all other cases, this shift is scalarized as the target does not support
; vpsllv instructions.

define <4 x i64> @test8(<4 x i64> %a) {
  %shl = shl <4 x i64> %a, <i64 1, i64 2, i64 3, i64 4>
  ret <4 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test8':
; SSE2: Found an estimated cost of 8 for instruction:   %shl
; SSE41: Found an estimated cost of 8 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; Same as 'test6', with the difference that the cost is double.

define <32 x i16> @test9(<32 x i16> %a) {
  %shl = shl <32 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <32 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test9':
; SSE2: Found an estimated cost of 4 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl


; Same as 'test7', except that now the cost is double.

define <16 x i32> @test10(<16 x i32> %a) {
  %shl = shl <16 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <16 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test10':
; SSE2: Found an estimated cost of 24 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a sequence of
; two vpsllvq instructions. Therefore, the expected cost is only 2.
; In all other cases, this shift is scalarized as we don't have vpsllv
; instructions.

define <8 x i64> @test11(<8 x i64> %a) {
  %shl = shl <8 x i64> %a, <i64 1, i64 1, i64 2, i64 3, i64 1, i64 1, i64 2, i64 3>
  ret <8 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test11':
; SSE2: Found an estimated cost of 16 for instruction:   %shl
; SSE41: Found an estimated cost of 16 for instruction:   %shl
; AVX: Found an estimated cost of 16 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl
