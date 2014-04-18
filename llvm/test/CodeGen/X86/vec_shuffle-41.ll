; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

; Use buildFromShuffleMostly which allows this to be generated as two 128-bit
; shuffles and an insert.

; This is the (somewhat questionable) LLVM IR that is generated for:
;    x8.s0123456 = x8.s1234567;  // x8 is a <8 x float> type
;    x8.s7 = f;                  // f is float


define <8 x float> @test1(<8 x float> %a, float %b) {
; CHECK-LABEL: test1:
; CHECK: vinsertps
; CHECK-NOT: vinsertps
entry:
  %shift = shufflevector <8 x float> %a, <8 x float> undef, <7 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %extend = shufflevector <7 x float> %shift, <7 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 undef>
  %insert = insertelement <8 x float> %extend, float %b, i32 7

  ret <8 x float> %insert
}
