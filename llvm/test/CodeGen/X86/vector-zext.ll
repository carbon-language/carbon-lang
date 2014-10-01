; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

define <8 x i32> @zext_8i16_to_8i32(<8 x i16> %A) nounwind uwtable readnone ssp {
; CHECK-LABEL: zext_8i16_to_8i32:
; CHECK:       ## BB#0: ## %entry
; CHECK-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; CHECK-NEXT:    vpunpckhwd {{.*#+}} xmm2 = xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; CHECK-NEXT:    vpunpcklwd {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; CHECK-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %B = zext <8 x i16> %A to <8 x i32>
  ret <8 x i32>%B
}

define <4 x i64> @zext_4i32_to_4i64(<4 x i32> %A) nounwind uwtable readnone ssp {
; CHECK-LABEL: zext_4i32_to_4i64:
; CHECK:       ## BB#0: ## %entry
; CHECK-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; CHECK-NEXT:    vpunpckhdq {{.*#+}} xmm2 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
; CHECK-NEXT:    vpunpckldq {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
; CHECK-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %B = zext <4 x i32> %A to <4 x i64>
  ret <4 x i64>%B
}

define <8 x i32> @zext_8i8_to_8i32(<8 x i8> %z) {
; CHECK-LABEL: zext_8i8_to_8i32:
; CHECK:       ## BB#0: ## %entry
; CHECK-NEXT:    vpunpckhwd {{.*#+}} xmm1 = xmm0[4,4,5,5,6,6,7,7]
; CHECK-NEXT:    vpmovzxwd %xmm0, %xmm0
; CHECK-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    vandps {{.*}}, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %t = zext <8 x i8> %z to <8 x i32>
  ret <8 x i32> %t
}

; PR17654
define <16 x i16> @zext_16i8_to_16i16(<16 x i8> %z) {
; CHECK-LABEL: zext_16i8_to_16i16:
; CHECK:       ## BB#0: ## %entry
; CHECK-NEXT:    vpxor %xmm1, %xmm1, %xmm1
; CHECK-NEXT:    vpunpckhbw {{.*#+}} xmm2 = xmm0[8],xmm1[8],xmm0[9],xmm1[9],xmm0[10],xmm1[10],xmm0[11],xmm1[11],xmm0[12],xmm1[12],xmm0[13],xmm1[13],xmm0[14],xmm1[14],xmm0[15],xmm1[15]
; CHECK-NEXT:    vpunpcklbw {{.*#+}} xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1],xmm0[2],xmm1[2],xmm0[3],xmm1[3],xmm0[4],xmm1[4],xmm0[5],xmm1[5],xmm0[6],xmm1[6],xmm0[7],xmm1[7]
; CHECK-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %t = zext <16 x i8> %z to <16 x i16>
  ret <16 x i16> %t
}
