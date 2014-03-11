; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; AVX2 Logical Shift Left

define <16 x i16> @test_sllw_1(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_1:
; CHECK: vpsllw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sllw_2(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_2:
; CHECK: vpaddw  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sllw_3(<16 x i16> %InVec) {
entry:
  %shl = shl <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sllw_3:
; CHECK: vpsllw $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_1(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_1:
; CHECK: vpslld  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_2(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_2:
; CHECK: vpaddd  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_slld_3(<8 x i32> %InVec) {
entry:
  %shl = shl <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_slld_3:
; CHECK: vpslld $31, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_1(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 0, i64 0, i64 0, i64 0>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_1:
; CHECK: vpsllq  $0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_2(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_2:
; CHECK: vpaddq  %ymm0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_sllq_3(<4 x i64> %InVec) {
entry:
  %shl = shl <4 x i64> %InVec, <i64 63, i64 63, i64 63, i64 63>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_sllq_3:
; CHECK: vpsllq $63, %ymm0, %ymm0
; CHECK: ret

; AVX2 Arithmetic Shift

define <16 x i16> @test_sraw_1(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_1:
; CHECK: vpsraw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sraw_2(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_2:
; CHECK: vpsraw  $1, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_sraw_3(<16 x i16> %InVec) {
entry:
  %shl = ashr <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_sraw_3:
; CHECK: vpsraw  $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_1(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_1:
; CHECK: vpsrad  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_2(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_2:
; CHECK: vpsrad  $1, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srad_3(<8 x i32> %InVec) {
entry:
  %shl = ashr <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srad_3:
; CHECK: vpsrad  $31, %ymm0, %ymm0
; CHECK: ret

; SSE Logical Shift Right

define <16 x i16> @test_srlw_1(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_1:
; CHECK: vpsrlw  $0, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_srlw_2(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_2:
; CHECK: vpsrlw  $1, %ymm0, %ymm0
; CHECK: ret

define <16 x i16> @test_srlw_3(<16 x i16> %InVec) {
entry:
  %shl = lshr <16 x i16> %InVec, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  ret <16 x i16> %shl
}

; CHECK-LABEL: test_srlw_3:
; CHECK: vpsrlw $15, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_1(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_1:
; CHECK: vpsrld  $0, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_2(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_2:
; CHECK: vpsrld  $1, %ymm0, %ymm0
; CHECK: ret

define <8 x i32> @test_srld_3(<8 x i32> %InVec) {
entry:
  %shl = lshr <8 x i32> %InVec, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  ret <8 x i32> %shl
}

; CHECK-LABEL: test_srld_3:
; CHECK: vpsrld $31, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_1(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 0, i64 0, i64 0, i64 0>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_1:
; CHECK: vpsrlq  $0, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_2(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 1, i64 1, i64 1, i64 1>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_2:
; CHECK: vpsrlq  $1, %ymm0, %ymm0
; CHECK: ret

define <4 x i64> @test_srlq_3(<4 x i64> %InVec) {
entry:
  %shl = lshr <4 x i64> %InVec, <i64 63, i64 63, i64 63, i64 63>
  ret <4 x i64> %shl
}

; CHECK-LABEL: test_srlq_3:
; CHECK: vpsrlq $63, %ymm0, %ymm0
; CHECK: ret
