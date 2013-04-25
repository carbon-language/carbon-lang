; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -mcpu=cortex-a9 | FileCheck %s

define <2 x i8> @sdiv_v2_i8(<2 x i8>  %a, <2 x i8> %b) {
  ; CHECK: sdiv_v2_i8
  ; CHECK: cost of 40 {{.*}} sdiv

  %1 = sdiv <2 x i8>  %a, %b
  ret <2 x i8> %1
}
define <2 x i16> @sdiv_v2_i16(<2 x i16>  %a, <2 x i16> %b) {
  ; CHECK: sdiv_v2_i16
  ; CHECK: cost of 40 {{.*}} sdiv

  %1 = sdiv <2 x i16>  %a, %b
  ret <2 x i16> %1
}
define <2 x i32> @sdiv_v2_i32(<2 x i32>  %a, <2 x i32> %b) {
  ; CHECK: sdiv_v2_i32
  ; CHECK: cost of 40 {{.*}} sdiv

  %1 = sdiv <2 x i32>  %a, %b
  ret <2 x i32> %1
}
define <2 x i64> @sdiv_v2_i64(<2 x i64>  %a, <2 x i64> %b) {
  ; CHECK: sdiv_v2_i64
  ; CHECK: cost of 40 {{.*}} sdiv

  %1 = sdiv <2 x i64>  %a, %b
  ret <2 x i64> %1
}
define <4 x i8> @sdiv_v4_i8(<4 x i8>  %a, <4 x i8> %b) {
  ; CHECK: sdiv_v4_i8
  ; CHECK: cost of 10 {{.*}} sdiv

  %1 = sdiv <4 x i8>  %a, %b
  ret <4 x i8> %1
}
define <4 x i16> @sdiv_v4_i16(<4 x i16>  %a, <4 x i16> %b) {
  ; CHECK: sdiv_v4_i16
  ; CHECK: cost of 10 {{.*}} sdiv

  %1 = sdiv <4 x i16>  %a, %b
  ret <4 x i16> %1
}
define <4 x i32> @sdiv_v4_i32(<4 x i32>  %a, <4 x i32> %b) {
  ; CHECK: sdiv_v4_i32
  ; CHECK: cost of 80 {{.*}} sdiv

  %1 = sdiv <4 x i32>  %a, %b
  ret <4 x i32> %1
}
define <4 x i64> @sdiv_v4_i64(<4 x i64>  %a, <4 x i64> %b) {
  ; CHECK: sdiv_v4_i64
  ; CHECK: cost of 80 {{.*}} sdiv

  %1 = sdiv <4 x i64>  %a, %b
  ret <4 x i64> %1
}
define <8 x i8> @sdiv_v8_i8(<8 x i8>  %a, <8 x i8> %b) {
  ; CHECK: sdiv_v8_i8
  ; CHECK: cost of 10 {{.*}} sdiv

  %1 = sdiv <8 x i8>  %a, %b
  ret <8 x i8> %1
}
define <8 x i16> @sdiv_v8_i16(<8 x i16>  %a, <8 x i16> %b) {
  ; CHECK: sdiv_v8_i16
  ; CHECK: cost of 160 {{.*}} sdiv

  %1 = sdiv <8 x i16>  %a, %b
  ret <8 x i16> %1
}
define <8 x i32> @sdiv_v8_i32(<8 x i32>  %a, <8 x i32> %b) {
  ; CHECK: sdiv_v8_i32
  ; CHECK: cost of 160 {{.*}} sdiv

  %1 = sdiv <8 x i32>  %a, %b
  ret <8 x i32> %1
}
define <8 x i64> @sdiv_v8_i64(<8 x i64>  %a, <8 x i64> %b) {
  ; CHECK: sdiv_v8_i64
  ; CHECK: cost of 160 {{.*}} sdiv

  %1 = sdiv <8 x i64>  %a, %b
  ret <8 x i64> %1
}
define <16 x i8> @sdiv_v16_i8(<16 x i8>  %a, <16 x i8> %b) {
  ; CHECK: sdiv_v16_i8
  ; CHECK: cost of 320 {{.*}} sdiv

  %1 = sdiv <16 x i8>  %a, %b
  ret <16 x i8> %1
}
define <16 x i16> @sdiv_v16_i16(<16 x i16>  %a, <16 x i16> %b) {
  ; CHECK: sdiv_v16_i16
  ; CHECK: cost of 320 {{.*}} sdiv

  %1 = sdiv <16 x i16>  %a, %b
  ret <16 x i16> %1
}
define <16 x i32> @sdiv_v16_i32(<16 x i32>  %a, <16 x i32> %b) {
  ; CHECK: sdiv_v16_i32
  ; CHECK: cost of 320 {{.*}} sdiv

  %1 = sdiv <16 x i32>  %a, %b
  ret <16 x i32> %1
}
define <16 x i64> @sdiv_v16_i64(<16 x i64>  %a, <16 x i64> %b) {
  ; CHECK: sdiv_v16_i64
  ; CHECK: cost of 320 {{.*}} sdiv

  %1 = sdiv <16 x i64>  %a, %b
  ret <16 x i64> %1
}
define <2 x i8> @udiv_v2_i8(<2 x i8>  %a, <2 x i8> %b) {
  ; CHECK: udiv_v2_i8
  ; CHECK: cost of 40 {{.*}} udiv

  %1 = udiv <2 x i8>  %a, %b
  ret <2 x i8> %1
}
define <2 x i16> @udiv_v2_i16(<2 x i16>  %a, <2 x i16> %b) {
  ; CHECK: udiv_v2_i16
  ; CHECK: cost of 40 {{.*}} udiv

  %1 = udiv <2 x i16>  %a, %b
  ret <2 x i16> %1
}
define <2 x i32> @udiv_v2_i32(<2 x i32>  %a, <2 x i32> %b) {
  ; CHECK: udiv_v2_i32
  ; CHECK: cost of 40 {{.*}} udiv

  %1 = udiv <2 x i32>  %a, %b
  ret <2 x i32> %1
}
define <2 x i64> @udiv_v2_i64(<2 x i64>  %a, <2 x i64> %b) {
  ; CHECK: udiv_v2_i64
  ; CHECK: cost of 40 {{.*}} udiv

  %1 = udiv <2 x i64>  %a, %b
  ret <2 x i64> %1
}
define <4 x i8> @udiv_v4_i8(<4 x i8>  %a, <4 x i8> %b) {
  ; CHECK: udiv_v4_i8
  ; CHECK: cost of 10 {{.*}} udiv

  %1 = udiv <4 x i8>  %a, %b
  ret <4 x i8> %1
}
define <4 x i16> @udiv_v4_i16(<4 x i16>  %a, <4 x i16> %b) {
  ; CHECK: udiv_v4_i16
  ; CHECK: cost of 10 {{.*}} udiv

  %1 = udiv <4 x i16>  %a, %b
  ret <4 x i16> %1
}
define <4 x i32> @udiv_v4_i32(<4 x i32>  %a, <4 x i32> %b) {
  ; CHECK: udiv_v4_i32
  ; CHECK: cost of 80 {{.*}} udiv

  %1 = udiv <4 x i32>  %a, %b
  ret <4 x i32> %1
}
define <4 x i64> @udiv_v4_i64(<4 x i64>  %a, <4 x i64> %b) {
  ; CHECK: udiv_v4_i64
  ; CHECK: cost of 80 {{.*}} udiv

  %1 = udiv <4 x i64>  %a, %b
  ret <4 x i64> %1
}
define <8 x i8> @udiv_v8_i8(<8 x i8>  %a, <8 x i8> %b) {
  ; CHECK: udiv_v8_i8
  ; CHECK: cost of 10 {{.*}} udiv

  %1 = udiv <8 x i8>  %a, %b
  ret <8 x i8> %1
}
define <8 x i16> @udiv_v8_i16(<8 x i16>  %a, <8 x i16> %b) {
  ; CHECK: udiv_v8_i16
  ; CHECK: cost of 160 {{.*}} udiv

  %1 = udiv <8 x i16>  %a, %b
  ret <8 x i16> %1
}
define <8 x i32> @udiv_v8_i32(<8 x i32>  %a, <8 x i32> %b) {
  ; CHECK: udiv_v8_i32
  ; CHECK: cost of 160 {{.*}} udiv

  %1 = udiv <8 x i32>  %a, %b
  ret <8 x i32> %1
}
define <8 x i64> @udiv_v8_i64(<8 x i64>  %a, <8 x i64> %b) {
  ; CHECK: udiv_v8_i64
  ; CHECK: cost of 160 {{.*}} udiv

  %1 = udiv <8 x i64>  %a, %b
  ret <8 x i64> %1
}
define <16 x i8> @udiv_v16_i8(<16 x i8>  %a, <16 x i8> %b) {
  ; CHECK: udiv_v16_i8
  ; CHECK: cost of 320 {{.*}} udiv

  %1 = udiv <16 x i8>  %a, %b
  ret <16 x i8> %1
}
define <16 x i16> @udiv_v16_i16(<16 x i16>  %a, <16 x i16> %b) {
  ; CHECK: udiv_v16_i16
  ; CHECK: cost of 320 {{.*}} udiv

  %1 = udiv <16 x i16>  %a, %b
  ret <16 x i16> %1
}
define <16 x i32> @udiv_v16_i32(<16 x i32>  %a, <16 x i32> %b) {
  ; CHECK: udiv_v16_i32
  ; CHECK: cost of 320 {{.*}} udiv

  %1 = udiv <16 x i32>  %a, %b
  ret <16 x i32> %1
}
define <16 x i64> @udiv_v16_i64(<16 x i64>  %a, <16 x i64> %b) {
  ; CHECK: udiv_v16_i64
  ; CHECK: cost of 320 {{.*}} udiv

  %1 = udiv <16 x i64>  %a, %b
  ret <16 x i64> %1
}
define <2 x i8> @srem_v2_i8(<2 x i8>  %a, <2 x i8> %b) {
  ; CHECK: srem_v2_i8
  ; CHECK: cost of 40 {{.*}} srem

  %1 = srem <2 x i8>  %a, %b
  ret <2 x i8> %1
}
define <2 x i16> @srem_v2_i16(<2 x i16>  %a, <2 x i16> %b) {
  ; CHECK: srem_v2_i16
  ; CHECK: cost of 40 {{.*}} srem

  %1 = srem <2 x i16>  %a, %b
  ret <2 x i16> %1
}
define <2 x i32> @srem_v2_i32(<2 x i32>  %a, <2 x i32> %b) {
  ; CHECK: srem_v2_i32
  ; CHECK: cost of 40 {{.*}} srem

  %1 = srem <2 x i32>  %a, %b
  ret <2 x i32> %1
}
define <2 x i64> @srem_v2_i64(<2 x i64>  %a, <2 x i64> %b) {
  ; CHECK: srem_v2_i64
  ; CHECK: cost of 40 {{.*}} srem

  %1 = srem <2 x i64>  %a, %b
  ret <2 x i64> %1
}
define <4 x i8> @srem_v4_i8(<4 x i8>  %a, <4 x i8> %b) {
  ; CHECK: srem_v4_i8
  ; CHECK: cost of 80 {{.*}} srem

  %1 = srem <4 x i8>  %a, %b
  ret <4 x i8> %1
}
define <4 x i16> @srem_v4_i16(<4 x i16>  %a, <4 x i16> %b) {
  ; CHECK: srem_v4_i16
  ; CHECK: cost of 80 {{.*}} srem

  %1 = srem <4 x i16>  %a, %b
  ret <4 x i16> %1
}
define <4 x i32> @srem_v4_i32(<4 x i32>  %a, <4 x i32> %b) {
  ; CHECK: srem_v4_i32
  ; CHECK: cost of 80 {{.*}} srem

  %1 = srem <4 x i32>  %a, %b
  ret <4 x i32> %1
}
define <4 x i64> @srem_v4_i64(<4 x i64>  %a, <4 x i64> %b) {
  ; CHECK: srem_v4_i64
  ; CHECK: cost of 80 {{.*}} srem

  %1 = srem <4 x i64>  %a, %b
  ret <4 x i64> %1
}
define <8 x i8> @srem_v8_i8(<8 x i8>  %a, <8 x i8> %b) {
  ; CHECK: srem_v8_i8
  ; CHECK: cost of 160 {{.*}} srem

  %1 = srem <8 x i8>  %a, %b
  ret <8 x i8> %1
}
define <8 x i16> @srem_v8_i16(<8 x i16>  %a, <8 x i16> %b) {
  ; CHECK: srem_v8_i16
  ; CHECK: cost of 160 {{.*}} srem

  %1 = srem <8 x i16>  %a, %b
  ret <8 x i16> %1
}
define <8 x i32> @srem_v8_i32(<8 x i32>  %a, <8 x i32> %b) {
  ; CHECK: srem_v8_i32
  ; CHECK: cost of 160 {{.*}} srem

  %1 = srem <8 x i32>  %a, %b
  ret <8 x i32> %1
}
define <8 x i64> @srem_v8_i64(<8 x i64>  %a, <8 x i64> %b) {
  ; CHECK: srem_v8_i64
  ; CHECK: cost of 160 {{.*}} srem

  %1 = srem <8 x i64>  %a, %b
  ret <8 x i64> %1
}
define <16 x i8> @srem_v16_i8(<16 x i8>  %a, <16 x i8> %b) {
  ; CHECK: srem_v16_i8
  ; CHECK: cost of 320 {{.*}} srem

  %1 = srem <16 x i8>  %a, %b
  ret <16 x i8> %1
}
define <16 x i16> @srem_v16_i16(<16 x i16>  %a, <16 x i16> %b) {
  ; CHECK: srem_v16_i16
  ; CHECK: cost of 320 {{.*}} srem

  %1 = srem <16 x i16>  %a, %b
  ret <16 x i16> %1
}
define <16 x i32> @srem_v16_i32(<16 x i32>  %a, <16 x i32> %b) {
  ; CHECK: srem_v16_i32
  ; CHECK: cost of 320 {{.*}} srem

  %1 = srem <16 x i32>  %a, %b
  ret <16 x i32> %1
}
define <16 x i64> @srem_v16_i64(<16 x i64>  %a, <16 x i64> %b) {
  ; CHECK: srem_v16_i64
  ; CHECK: cost of 320 {{.*}} srem

  %1 = srem <16 x i64>  %a, %b
  ret <16 x i64> %1
}
define <2 x i8> @urem_v2_i8(<2 x i8>  %a, <2 x i8> %b) {
  ; CHECK: urem_v2_i8
  ; CHECK: cost of 40 {{.*}} urem

  %1 = urem <2 x i8>  %a, %b
  ret <2 x i8> %1
}
define <2 x i16> @urem_v2_i16(<2 x i16>  %a, <2 x i16> %b) {
  ; CHECK: urem_v2_i16
  ; CHECK: cost of 40 {{.*}} urem

  %1 = urem <2 x i16>  %a, %b
  ret <2 x i16> %1
}
define <2 x i32> @urem_v2_i32(<2 x i32>  %a, <2 x i32> %b) {
  ; CHECK: urem_v2_i32
  ; CHECK: cost of 40 {{.*}} urem

  %1 = urem <2 x i32>  %a, %b
  ret <2 x i32> %1
}
define <2 x i64> @urem_v2_i64(<2 x i64>  %a, <2 x i64> %b) {
  ; CHECK: urem_v2_i64
  ; CHECK: cost of 40 {{.*}} urem

  %1 = urem <2 x i64>  %a, %b
  ret <2 x i64> %1
}
define <4 x i8> @urem_v4_i8(<4 x i8>  %a, <4 x i8> %b) {
  ; CHECK: urem_v4_i8
  ; CHECK: cost of 80 {{.*}} urem

  %1 = urem <4 x i8>  %a, %b
  ret <4 x i8> %1
}
define <4 x i16> @urem_v4_i16(<4 x i16>  %a, <4 x i16> %b) {
  ; CHECK: urem_v4_i16
  ; CHECK: cost of 80 {{.*}} urem

  %1 = urem <4 x i16>  %a, %b
  ret <4 x i16> %1
}
define <4 x i32> @urem_v4_i32(<4 x i32>  %a, <4 x i32> %b) {
  ; CHECK: urem_v4_i32
  ; CHECK: cost of 80 {{.*}} urem

  %1 = urem <4 x i32>  %a, %b
  ret <4 x i32> %1
}
define <4 x i64> @urem_v4_i64(<4 x i64>  %a, <4 x i64> %b) {
  ; CHECK: urem_v4_i64
  ; CHECK: cost of 80 {{.*}} urem

  %1 = urem <4 x i64>  %a, %b
  ret <4 x i64> %1
}
define <8 x i8> @urem_v8_i8(<8 x i8>  %a, <8 x i8> %b) {
  ; CHECK: urem_v8_i8
  ; CHECK: cost of 160 {{.*}} urem

  %1 = urem <8 x i8>  %a, %b
  ret <8 x i8> %1
}
define <8 x i16> @urem_v8_i16(<8 x i16>  %a, <8 x i16> %b) {
  ; CHECK: urem_v8_i16
  ; CHECK: cost of 160 {{.*}} urem

  %1 = urem <8 x i16>  %a, %b
  ret <8 x i16> %1
}
define <8 x i32> @urem_v8_i32(<8 x i32>  %a, <8 x i32> %b) {
  ; CHECK: urem_v8_i32
  ; CHECK: cost of 160 {{.*}} urem

  %1 = urem <8 x i32>  %a, %b
  ret <8 x i32> %1
}
define <8 x i64> @urem_v8_i64(<8 x i64>  %a, <8 x i64> %b) {
  ; CHECK: urem_v8_i64
  ; CHECK: cost of 160 {{.*}} urem

  %1 = urem <8 x i64>  %a, %b
  ret <8 x i64> %1
}
define <16 x i8> @urem_v16_i8(<16 x i8>  %a, <16 x i8> %b) {
  ; CHECK: urem_v16_i8
  ; CHECK: cost of 320 {{.*}} urem

  %1 = urem <16 x i8>  %a, %b
  ret <16 x i8> %1
}
define <16 x i16> @urem_v16_i16(<16 x i16>  %a, <16 x i16> %b) {
  ; CHECK: urem_v16_i16
  ; CHECK: cost of 320 {{.*}} urem

  %1 = urem <16 x i16>  %a, %b
  ret <16 x i16> %1
}
define <16 x i32> @urem_v16_i32(<16 x i32>  %a, <16 x i32> %b) {
  ; CHECK: urem_v16_i32
  ; CHECK: cost of 320 {{.*}} urem

  %1 = urem <16 x i32>  %a, %b
  ret <16 x i32> %1
}
define <16 x i64> @urem_v16_i64(<16 x i64>  %a, <16 x i64> %b) {
  ; CHECK: urem_v16_i64
  ; CHECK: cost of 320 {{.*}} urem

  %1 = urem <16 x i64>  %a, %b
  ret <16 x i64> %1
}
