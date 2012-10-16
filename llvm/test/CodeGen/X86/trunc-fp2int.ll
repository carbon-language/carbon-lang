; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7 | FileCheck %s

define <4 x i8> @bar(<4 x float> %in) nounwind readnone alwaysinline {
  %1 = fptoui <4 x float> %in to <4 x i8>
  ret <4 x i8> %1
; CHECK: bar
; CHECK: cvttps2dq
}
define <4 x i8> @foo(<4 x float> %in) nounwind readnone alwaysinline {
  %1 = fptoui <4 x float> %in to <4 x i32>
  %2 = trunc <4 x i32> %1 to <4 x i16>
  %3 = shufflevector <4 x i16> %2, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %4 = trunc <8 x i16> %3 to <8 x i8>
  %5 = shufflevector <8 x i8> %4, <8 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i8> %5
; CHECK: foo
; CHECK: cvttps2dq
}
