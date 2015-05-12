;RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s
;RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck --check-prefix=SKX %s

;CHECK-LABEL: shift_16_i32
;CHECK: vpsrld
;CHECK: vpslld
;CHECK: vpsrad
;CHECK: ret
define <16 x i32> @shift_16_i32(<16 x i32> %a) {
   %b = lshr <16 x i32> %a, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
   %c = shl <16 x i32> %b, <i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
   %d = ashr <16 x i32> %c, <i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12, i32 12>
   ret <16 x i32> %d;
}

;CHECK-LABEL: shift_8_i64
;CHECK: vpsrlq
;CHECK: vpsllq
;CHECK: vpsraq
;CHECK: ret
define <8 x i64> @shift_8_i64(<8 x i64> %a) {
   %b = lshr <8 x i64> %a, <i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1>
   %c = shl <8 x i64> %b,  <i64 12, i64 12, i64 12, i64 12, i64 12, i64 12, i64 12, i64 12>
   %d = ashr <8 x i64> %c, <i64 12, i64 12, i64 12, i64 12, i64 12, i64 12, i64 12, i64 12>
   ret <8 x i64> %d;
}

;SKX-LABEL: shift_4_i64
;SKX: vpsrlq
;SKX: vpsllq
;SKX: vpsraq
;SKX: ret
define <4 x i64> @shift_4_i64(<4 x i64> %a) {
   %b = lshr <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
   %c = shl <4 x i64> %b,  <i64 12, i64 12, i64 12, i64 12>
   %d = ashr <4 x i64> %c, <i64 12, i64 12, i64 12, i64 12>
   ret <4 x i64> %d;
}

; CHECK-LABEL: variable_shl4
; CHECK: vpsllvq %zmm
; CHECK: ret
define <8 x i64> @variable_shl4(<8 x i64> %x, <8 x i64> %y) {
  %k = shl <8 x i64> %x, %y
  ret <8 x i64> %k
}

; CHECK-LABEL: variable_shl5
; CHECK: vpsllvd %zmm
; CHECK: ret
define <16 x i32> @variable_shl5(<16 x i32> %x, <16 x i32> %y) {
  %k = shl <16 x i32> %x, %y
  ret <16 x i32> %k
}

; CHECK-LABEL: variable_srl0
; CHECK: vpsrlvd
; CHECK: ret
define <16 x i32> @variable_srl0(<16 x i32> %x, <16 x i32> %y) {
  %k = lshr <16 x i32> %x, %y
  ret <16 x i32> %k
}

; CHECK-LABEL: variable_srl2
; CHECK: psrlvq
; CHECK: ret
define <8 x i64> @variable_srl2(<8 x i64> %x, <8 x i64> %y) {
  %k = lshr <8 x i64> %x, %y
  ret <8 x i64> %k
}

; CHECK-LABEL: variable_sra1
; CHECK: vpsravd
; CHECK: ret
define <16 x i32> @variable_sra1(<16 x i32> %x, <16 x i32> %y) {
  %k = ashr <16 x i32> %x, %y
  ret <16 x i32> %k
}

; CHECK-LABEL: variable_sra2
; CHECK: vpsravq %zmm
; CHECK: ret
define <8 x i64> @variable_sra2(<8 x i64> %x, <8 x i64> %y) {
  %k = ashr <8 x i64> %x, %y
  ret <8 x i64> %k
}

; SKX-LABEL: variable_sra3
; SKX: vpsravq %ymm
; SKX: ret
define <4 x i64> @variable_sra3(<4 x i64> %x, <4 x i64> %y) {
  %k = ashr <4 x i64> %x, %y
  ret <4 x i64> %k
}

; SKX-LABEL: variable_sra4
; SKX: vpsravw %xmm
; SKX: ret
define <8 x i16> @variable_sra4(<8 x i16> %x, <8 x i16> %y) {
  %k = ashr <8 x i16> %x, %y
  ret <8 x i16> %k
}

; CHECK-LABEL: variable_sra01_load
; CHECK: vpsravd (%
; CHECK: ret
define <16 x i32> @variable_sra01_load(<16 x i32> %x, <16 x i32>* %y) {
  %y1 = load <16 x i32>, <16 x i32>* %y
  %k = ashr <16 x i32> %x, %y1
  ret <16 x i32> %k
}

; CHECK-LABEL: variable_shl1_load
; CHECK: vpsllvd (%
; CHECK: ret
define <16 x i32> @variable_shl1_load(<16 x i32> %x, <16 x i32>* %y) {
  %y1 = load <16 x i32>, <16 x i32>* %y
  %k = shl <16 x i32> %x, %y1
  ret <16 x i32> %k
}
; CHECK: variable_srl0_load
; CHECK: vpsrlvd (%
; CHECK: ret
define <16 x i32> @variable_srl0_load(<16 x i32> %x, <16 x i32>* %y) {
  %y1 = load <16 x i32>, <16 x i32>* %y
  %k = lshr <16 x i32> %x, %y1
  ret <16 x i32> %k
}

; CHECK: variable_srl3_load
; CHECK: vpsrlvq (%
; CHECK: ret
define <8 x i64> @variable_srl3_load(<8 x i64> %x, <8 x i64>* %y) {
  %y1 = load <8 x i64>, <8 x i64>* %y
  %k = lshr <8 x i64> %x, %y1
  ret <8 x i64> %k
}
