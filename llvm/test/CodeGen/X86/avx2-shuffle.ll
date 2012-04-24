; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; Make sure that we don't match this shuffle using the vpblendw instruction. 
; The mask for the vpblendw instruction needs to be identical for both halves 
; of the YMM.

; CHECK: blendw1
; CHECK-NOT: vpblendw
; CHECK: ret
define <16 x i16> @blendw1(<16 x i16> %a, <16 x i16> %b) nounwind alwaysinline {
  %t = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 18, i32 3, i32 20, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %t
}
