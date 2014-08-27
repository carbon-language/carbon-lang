; RUN: llc < %s -march=x86-64 -mattr=+avx | FileCheck %s
; RUN: llc < %s -march=x86 -mattr=+avx | FileCheck %s

define<4 x i32> @func_16_32() {
  %F = load <4 x i16>* undef
  %G = sext <4 x i16> %F to <4 x i32>
  %H = load <4 x i16>* undef
  %Y = sext <4 x i16> %H to <4 x i32>
  %T = add <4 x i32> %Y, %G
  store <4 x i32>%T , <4 x i32>* undef
  ret <4 x i32> %T
}

define<4 x i64> @func_16_64() {
  %F = load <4 x i16>* undef
  %G = sext <4 x i16> %F to <4 x i64>
  %H = load <4 x i16>* undef
  %Y = sext <4 x i16> %H to <4 x i64>
  %T = xor <4 x i64> %Y, %G
  store <4 x i64>%T , <4 x i64>* undef
  ret <4 x i64> %T
}

define<4 x i64> @func_32_64() {
  %F = load <4 x i32>* undef
  %G = sext <4 x i32> %F to <4 x i64>
  %H = load <4 x i32>* undef
  %Y = sext <4 x i32> %H to <4 x i64>
  %T = or <4 x i64> %Y, %G
  ret <4 x i64> %T
}

define<4 x i16> @func_8_16() {
  %F = load <4 x i8>* undef
  %G = sext <4 x i8> %F to <4 x i16>
  %H = load <4 x i8>* undef
  %Y = sext <4 x i8> %H to <4 x i16>
  %T = add <4 x i16> %Y, %G
  ret <4 x i16> %T
}

define<4 x i32> @func_8_32() {
  %F = load <4 x i8>* undef
  %G = sext <4 x i8> %F to <4 x i32>
  %H = load <4 x i8>* undef
  %Y = sext <4 x i8> %H to <4 x i32>
  %T = sub <4 x i32> %Y, %G
  ret <4 x i32> %T
}

define<4 x i64> @func_8_64() {
  %F = load <4 x i8>* undef
  %G = sext <4 x i8> %F to <4 x i64>
  %H = load <4 x i8>* undef
  %Y = sext <4 x i8> %H to <4 x i64>
  %T = add <4 x i64> %Y, %G
  ret <4 x i64> %T
}

define<4 x i32> @const_16_32() {
  %G = sext <4 x i16> <i16 0, i16 3, i16 8, i16 7> to <4 x i32>
  ret <4 x i32> %G
}

define<4 x i64> @const_16_64() {
  %G = sext <4 x i16> <i16 0, i16 3, i16 8, i16 7> to <4 x i64>
  ret <4 x i64> %G
}

define <4 x i32> @sextload(<4 x i16>* %ptr) {
; From PR20767 - make sure that we correctly use SSE4.1 to do sign extension
; loads for both 32-bit and 64-bit x86 targets.
; CHECK-LABEL: sextload:
; CHECK:         vpmovsxwd {{.*}}, %xmm0
; CHECK-NEXT:    ret
entry:
  %l = load<4 x i16>* %ptr
  %m = sext<4 x i16> %l to <4 x i32>
  ret <4 x i32> %m
}
