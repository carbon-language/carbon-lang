; Test vector register moves.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 moves.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <16 x i8> %val2
}

; Test v8i16 moves.
define <8 x i16> @f2(<8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <8 x i16> %val2
}

; Test v4i32 moves.
define <4 x i32> @f3(<4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <4 x i32> %val2
}

; Test v2i64 moves.
define <2 x i64> @f4(<2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <2 x i64> %val2
}

; Test v4f32 moves.
define <4 x float> @f5(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f5:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <4 x float> %val2
}

; Test v2f64 moves.
define <2 x double> @f6(<2 x double> %val1, <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: vlr %v24, %v26
; CHECK: br %r14
  ret <2 x double> %val2
}
