; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpaddq %ymm
define <4 x i64> @vpaddq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = add <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK: vpaddd %ymm
define <8 x i32> @vpaddd(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = add <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpaddw %ymm
define <16 x i16> @vpaddw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = add <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpaddb %ymm
define <32 x i8> @vpaddb(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = add <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK: vpsubq %ymm
define <4 x i64> @vpsubq(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = sub <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK: vpsubd %ymm
define <8 x i32> @vpsubd(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = sub <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpsubw %ymm
define <16 x i16> @vpsubw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = sub <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpsubb %ymm
define <32 x i8> @vpsubb(<32 x i8> %i, <32 x i8> %j) nounwind readnone {
  %x = sub <32 x i8> %i, %j
  ret <32 x i8> %x
}

; CHECK: vpmulld %ymm
define <8 x i32> @vpmulld(<8 x i32> %i, <8 x i32> %j) nounwind readnone {
  %x = mul <8 x i32> %i, %j
  ret <8 x i32> %x
}

; CHECK: vpmullw %ymm
define <16 x i16> @vpmullw(<16 x i16> %i, <16 x i16> %j) nounwind readnone {
  %x = mul <16 x i16> %i, %j
  ret <16 x i16> %x
}

; CHECK: vpmuludq %ymm
; CHECK-NEXT: vpsrlq $32, %ymm
; CHECK-NEXT: vpmuludq %ymm
; CHECK-NEXT: vpsllq $32, %ymm
; CHECK-NEXT: vpaddq %ymm
; CHECK-NEXT: vpsrlq $32, %ymm
; CHECK-NEXT: vpmuludq %ymm
; CHECK-NEXT: vpsllq $32, %ymm
; CHECK-NEXT: vpaddq %ymm
define <4 x i64> @mul-v4i64(<4 x i64> %i, <4 x i64> %j) nounwind readnone {
  %x = mul <4 x i64> %i, %j
  ret <4 x i64> %x
}

; CHECK: mul_const1
; CHECK: vpaddd
; CHECK: ret
define <8 x i32> @mul_const1(<8 x i32> %x) {
  %y = mul <8 x i32> %x, <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x i32> %y
}

; CHECK: mul_const2
; CHECK: vpsllq  $2
; CHECK: ret
define <4 x i64> @mul_const2(<4 x i64> %x) {
  %y = mul <4 x i64> %x, <i64 4, i64 4, i64 4, i64 4>
  ret <4 x i64> %y
}

; CHECK: mul_const3
; CHECK: vpsllw  $3
; CHECK: ret
define <16 x i16> @mul_const3(<16 x i16> %x) {
  %y = mul <16 x i16> %x, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <16 x i16> %y
}

; CHECK: mul_const4
; CHECK: vpxor
; CHECK: vpsubq
; CHECK: ret
define <4 x i64> @mul_const4(<4 x i64> %x) {
  %y = mul <4 x i64> %x, <i64 -1, i64 -1, i64 -1, i64 -1>
  ret <4 x i64> %y
}

; CHECK: mul_const5
; CHECK: vxorps
; CHECK-NEXT: ret
define <8 x i32> @mul_const5(<8 x i32> %x) {
  %y = mul <8 x i32> %x, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %y
}

; CHECK: mul_const6
; CHECK: vpmulld
; CHECK: ret
define <8 x i32> @mul_const6(<8 x i32> %x) {
  %y = mul <8 x i32> %x, <i32 0, i32 0, i32 0, i32 2, i32 0, i32 2, i32 0, i32 0>
  ret <8 x i32> %y
}

; CHECK: mul_const7
; CHECK: vpaddq
; CHECK: vpaddq
; CHECK: ret
define <8 x i64> @mul_const7(<8 x i64> %x) {
  %y = mul <8 x i64> %x, <i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2, i64 2>
  ret <8 x i64> %y
}

; CHECK: mul_const8
; CHECK: vpsllw  $3
; CHECK: ret
define <8 x i16> @mul_const8(<8 x i16> %x) {
  %y = mul <8 x i16> %x, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %y
}

; CHECK: mul_const9
; CHECK: vpmulld
; CHECK: ret
define <8 x i32> @mul_const9(<8 x i32> %x) {
  %y = mul <8 x i32> %x, <i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %y
}
