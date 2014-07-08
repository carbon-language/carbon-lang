; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 | FileCheck %s

; Test that we correctly fold a shuffle that performs a swizzle of another
; shuffle node according to the rule
;  shuffle (shuffle (x, undef, M0), undef, M1) -> shuffle(x, undef, M2)
;
; We only do this if the resulting mask is legal to avoid introducing an
; illegal shuffle that is expanded into a sub-optimal sequence of instructions
; during lowering stage.

; Check that we produce a single vector permute / shuffle in all cases.

define <8 x i32> @swizzle_1(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 3, i32 1, i32 2, i32 0, i32 7, i32 5, i32 6, i32 4>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 1, i32 0, i32 2, i32 3, i32 7, i32 5, i32 6, i32 4>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_1
; CHECK: vpermd
; CHECK-NOT: vpermd
; CHECK: ret


define <8 x i32> @swizzle_2(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 6, i32 7, i32 4, i32 5, i32 0, i32 1, i32 2, i32 3>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 6, i32 7, i32 4, i32 5, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_2
; CHECK: vpshufd $78
; CHECK-NOT: vpermd
; CHECK-NOT: vpshufd
; CHECK: ret


define <8 x i32> @swizzle_3(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 2, i32 3, i32 0, i32 1>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 2, i32 3, i32 0, i32 1>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_3
; CHECK: vpshufd $78
; CHECK-NOT: vpermd
; CHECK-NOT: vpshufd
; CHECK: ret


define <8 x i32> @swizzle_4(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 4, i32 7, i32 5, i32 6, i32 3, i32 2, i32 0, i32 1>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 4, i32 7, i32 5, i32 6, i32 3, i32 2, i32 0, i32 1>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_4
; CHECK: vpermd
; CHECK-NOT: vpermd
; CHECK: ret


define <8 x i32> @swizzle_5(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 7, i32 4, i32 6, i32 5, i32 0, i32 2, i32 1, i32 3>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 7, i32 4, i32 6, i32 5, i32 0, i32 2, i32 1, i32 3>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_5
; CHECK: vpermd
; CHECK-NOT: vpermd
; CHECK: ret


define <8 x i32> @swizzle_6(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 2, i32 1, i32 3, i32 0, i32 4, i32 7, i32 6, i32 5>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 2, i32 1, i32 3, i32 0, i32 4, i32 7, i32 6, i32 5>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_6
; CHECK: vpermd
; CHECK-NOT: vpermd
; CHECK: ret


define <8 x i32> @swizzle_7(<8 x i32> %v) {
  %1 = shufflevector <8 x i32> %v, <8 x i32> undef, <8 x i32> <i32 0, i32 3, i32 1, i32 2, i32 5, i32 4, i32 6, i32 7>
  %2 = shufflevector <8 x i32> %1, <8 x i32> undef, <8 x i32> <i32 0, i32 3, i32 1, i32 2, i32 5, i32 4, i32 6, i32 7>
  ret <8 x i32> %2
}
; CHECK-LABEL: swizzle_7
; CHECK: vpermd
; CHECK-NOT: vpermd
; CHECK: ret


