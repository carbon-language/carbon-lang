; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; Make sure that we don't match this shuffle using the vpblendw YMM instruction.
; The mask for the vpblendw instruction needs to be identical for both halves
; of the YMM. Need to use two vpblendw instructions.

; CHECK: vpblendw_test1
; mask = 10010110,b = 150,d
; CHECK: vpblendw  $150, %ymm
; CHECK: ret
define <16 x i16> @vpblendw_test1(<16 x i16> %a, <16 x i16> %b) nounwind alwaysinline {
  %t = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 18, i32 3,  i32 20, i32 5,  i32 6,  i32 23, 
                                                               i32 8, i32 25, i32 26, i32 11, i32 28, i32 13, i32 14, i32 31>
  ret <16 x i16> %t
}

; CHECK: vpblendw_test2
; mask1 = 00010110 = 22
; mask2 = 10000000 = 128
; CHECK: vpblendw  $128, %xmm
; CHECK: vpblendw  $22, %xmm
; CHECK: vinserti128
; CHECK: ret
define <16 x i16> @vpblendw_test2(<16 x i16> %a, <16 x i16> %b) nounwind alwaysinline {
  %t = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 18, i32 3, i32 20, i32 5, i32 6, i32 7, 
                                                               i32 8, i32 9,  i32 10, i32 11, i32 12, i32 13, i32 14, i32 31>
  ret <16 x i16> %t
}

; CHECK: blend_test1
; CHECK: vpblendd
; CHECK: ret
define <8 x i32> @blend_test1(<8 x i32> %a, <8 x i32> %b) nounwind alwaysinline {
  %t = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 3, i32 12, i32 5, i32 6, i32 7>
  ret <8 x i32> %t
}

; CHECK: blend_test2
; CHECK: vpblendd
; CHECK: ret
define <8 x i32> @blend_test2(<8 x i32> %a, <8 x i32> %b) nounwind alwaysinline {
  %t = shufflevector <8 x i32> %a, <8 x i32> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 3, i32 12, i32 5, i32 6, i32 7>
  ret <8 x i32> %t
}


; CHECK: blend_test3
; CHECK: vblendps
; CHECK: ret
define <8 x float> @blend_test3(<8 x float> %a, <8 x float> %b) nounwind alwaysinline {
  %t = shufflevector <8 x float> %a, <8 x float> %b, <8 x i32> <i32 0, i32 9, i32 10, i32 3, i32 12, i32 5, i32 6, i32 7>
  ret <8 x float> %t
}

; CHECK: blend_test4
; CHECK: vblendpd
; CHECK: ret
define <4 x i64> @blend_test4(<4 x i64> %a, <4 x i64> %b) nounwind alwaysinline {
  %t = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 0, i32 5, i32 6, i32 3>
  ret <4 x i64> %t
}

;; 2 tests for shufflevectors that optimize to blend + immediate
; CHECK-LABEL: @blend_test5
; CHECK: vpblendd
; CHECK: ret
define <4 x i32> @blend_test5(<4 x i32> %a, <4 x i32> %b) {
  %1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret <4 x i32> %1
}

; CHECK-LABEL: @blend_test6
; CHECK: vpblendw
; CHECK: ret
define <16 x i16> @blend_test6(<16 x i16> %a, <16 x i16> %b) {
  %1 = shufflevector <16 x i16> %a, <16 x i16> %b, <16 x i32> <i32 0, i32 17, i32 18, i32  3, i32  4, i32  5, i32  6, i32 23,
                                                               i32 8, i32 25, i32 26, i32 11, i32 12, i32 13, i32 14, i32 31>
  ret <16 x i16> %1
}

; CHECK: vpshufhw $27, %ymm
define <16 x i16> @vpshufhw(<16 x i16> %src1) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <16 x i16> %src1, <16 x i16> %src1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 5, i32 4, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i16> %shuffle.i
}

; CHECK: vpshuflw $27, %ymm
define <16 x i16> @vpshuflw(<16 x i16> %src1) nounwind uwtable readnone ssp {
entry:
  %shuffle.i = shufflevector <16 x i16> %src1, <16 x i16> %src1, <16 x i32> <i32 3, i32 undef, i32 1, i32 0, i32 4, i32 5, i32 6, i32 7, i32 11, i32 10, i32 9, i32 8, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i16> %shuffle.i
}

; CHECK: vpshufb_test
; CHECK: vpshufb {{.*\(%r.*}}, %ymm
; CHECK: ret
define <32 x i8> @vpshufb_test(<32 x i8> %a) nounwind {
  %S = shufflevector <32 x i8> %a, <32 x i8> undef, <32 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15, 
                                                                i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15,  
                                                                i32 18, i32 19, i32 30, i32 16, i32 25, i32 23, i32 17, i32 25, 
                                                                i32 20, i32 19, i32 31, i32 17, i32 23, i32 undef, i32 29, i32 18>
  ret <32 x i8>%S
}

; CHECK: vpshufb1_test
; CHECK: vpshufb {{.*\(%r.*}}, %ymm
; CHECK: ret
define <32 x i8> @vpshufb1_test(<32 x i8> %a) nounwind {
  %S = shufflevector <32 x i8> %a, <32 x i8> zeroinitializer, <32 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15, 
                                                                i32 1, i32 9, i32 36, i32 11, i32 5, i32 13, i32 7, i32 15,  
                                                                i32 18, i32 49, i32 30, i32 16, i32 25, i32 23, i32 17, i32 25, 
                                                                i32 20, i32 19, i32 31, i32 17, i32 23, i32 undef, i32 29, i32 18>
  ret <32 x i8>%S
}


; CHECK: vpshufb2_test
; CHECK: vpshufb {{.*\(%r.*}}, %ymm
; CHECK: ret
define <32 x i8> @vpshufb2_test(<32 x i8> %a) nounwind {
  %S = shufflevector <32 x i8> zeroinitializer, <32 x i8> %a, <32 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15, 
                                                                i32 1, i32 9, i32 36, i32 11, i32 5, i32 13, i32 7, i32 15,  
                                                                i32 18, i32 49, i32 30, i32 16, i32 25, i32 23, i32 17, i32 25, 
                                                                i32 20, i32 19, i32 31, i32 17, i32 23, i32 undef, i32 29, i32 18>
  ret <32 x i8>%S
}
