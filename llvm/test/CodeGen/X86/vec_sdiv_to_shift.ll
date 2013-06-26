; RUN: llc < %s -march=x86-64 -mcpu=penryn -mattr=+avx2 | FileCheck %s


define <8 x i16> @sdiv_vec8x16(<8 x i16> %var) {
entry:
; CHECK: sdiv_vec8x16
; CHECK: psraw  $15
; CHECK: vpsrlw  $11
; CHECK: vpaddw
; CHECK: vpsraw  $5
; CHECK: ret
  %0 = sdiv <8 x i16> %var, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %0
}

define <4 x i32> @sdiv_zero(<4 x i32> %var) {
entry:
; CHECK: sdiv_zero
; CHECK-NOT: sra
; CHECK: ret
  %0 = sdiv <4 x i32> %var, <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %0
}

define <4 x i32> @sdiv_vec4x32(<4 x i32> %var) {
entry:
; CHECK: sdiv_vec4x32
; CHECK: vpsrad $31
; CHECK: vpsrld $28
; CHECK: vpaddd
; CHECK: vpsrad $4
; CHECK: ret
%0 = sdiv <4 x i32> %var, <i32 16, i32 16, i32 16, i32 16>
ret <4 x i32> %0
}

define <4 x i32> @sdiv_negative(<4 x i32> %var) {
entry:
; CHECK: sdiv_negative
; CHECK: vpsrad $31
; CHECK: vpsrld $28
; CHECK: vpaddd
; CHECK: vpsrad $4
; CHECK: vpsubd
; CHECK: ret
%0 = sdiv <4 x i32> %var, <i32 -16, i32 -16, i32 -16, i32 -16>
ret <4 x i32> %0
}

define <8 x i32> @sdiv8x32(<8 x i32> %var) {
entry:
; CHECK: sdiv8x32
; CHECK: vpsrad $31
; CHECK: vpsrld $26
; CHECK: vpaddd
; CHECK: vpsrad $6
; CHECK: ret
%0 = sdiv <8 x i32> %var, <i32 64, i32 64, i32 64, i32 64, i32 64, i32 64, i32 64, i32 64>
ret <8 x i32> %0
}

define <16 x i16> @sdiv16x16(<16 x i16> %var) {
entry:
; CHECK: sdiv16x16
; CHECK: vpsraw  $15
; CHECK: vpsrlw  $14
; CHECK: vpaddw
; CHECK: vpsraw  $2
; CHECK: ret
  %a0 = sdiv <16 x i16> %var, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  ret <16 x i16> %a0
}

; CHECK: sdiv_non_splat
; CHECK: idivl
; CHECK: ret
define <4 x i32> @sdiv_non_splat(<4 x i32> %x) {
  %y = sdiv <4 x i32> %x, <i32 2, i32 0, i32 0, i32 0>
  ret <4 x i32> %y
}