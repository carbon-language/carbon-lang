; RUN: llc < %s -mtriple=x86_64-pc-linux -mattr=+sse2 -mcpu=corei7 | FileCheck %s

; SSE2 Logical Shift Left

define <8 x i16> @test_sllw_1(<8 x i16> %InVec) {
entry:
  %shl = shl <8 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <8 x i16> %shl
}

; CHECK: test_sllw_1:
; CHECK: psllw   $0, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_sllw_2(<8 x i16> %InVec) {
entry:
  %shl = shl <8 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %shl
}

; CHECK: test_sllw_2:
; CHECK: paddw   %xmm0, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_sllw_3(<8 x i16> %InVec) {
entry:
  %shl = shl <8 x i16> %InVec, <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
  ret <8 x i16> %shl
}

; CHECK: test_sllw_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_slld_1(<4 x i32> %InVec) {
entry:
  %shl = shl <4 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %shl
}

; CHECK: test_slld_1:
; CHECK: pslld   $0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_slld_2(<4 x i32> %InVec) {
entry:
  %shl = shl <4 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shl
}

; CHECK: test_slld_2:
; CHECK: paddd   %xmm0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_slld_3(<4 x i32> %InVec) {
entry:
  %shl = shl <4 x i32> %InVec, <i32 32, i32 32, i32 32, i32 32>
  ret <4 x i32> %shl
}

; CHECK: test_slld_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_sllq_1(<2 x i64> %InVec) {
entry:
  %shl = shl <2 x i64> %InVec, <i64 0, i64 0>
  ret <2 x i64> %shl
}

; CHECK: test_sllq_1:
; CHECK: psllq   $0, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_sllq_2(<2 x i64> %InVec) {
entry:
  %shl = shl <2 x i64> %InVec, <i64 1, i64 1>
  ret <2 x i64> %shl
}

; CHECK: test_sllq_2:
; CHECK: paddq   %xmm0, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_sllq_3(<2 x i64> %InVec) {
entry:
  %shl = shl <2 x i64> %InVec, <i64 64, i64 64>
  ret <2 x i64> %shl
}

; CHECK: test_sllq_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret

; SSE2 Arithmetic Shift

define <8 x i16> @test_sraw_1(<8 x i16> %InVec) {
entry:
  %shl = ashr <8 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <8 x i16> %shl
}

; CHECK: test_sraw_1:
; CHECK: psraw   $0, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_sraw_2(<8 x i16> %InVec) {
entry:
  %shl = ashr <8 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %shl
}

; CHECK: test_sraw_2:
; CHECK: psraw   $1, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_sraw_3(<8 x i16> %InVec) {
entry:
  %shl = ashr <8 x i16> %InVec, <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
  ret <8 x i16> %shl
}

; CHECK: test_sraw_3:
; CHECK: psraw   $16, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srad_1(<4 x i32> %InVec) {
entry:
  %shl = ashr <4 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %shl
}

; CHECK: test_srad_1:
; CHECK: psrad   $0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srad_2(<4 x i32> %InVec) {
entry:
  %shl = ashr <4 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shl
}

; CHECK: test_srad_2:
; CHECK: psrad   $1, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srad_3(<4 x i32> %InVec) {
entry:
  %shl = ashr <4 x i32> %InVec, <i32 32, i32 32, i32 32, i32 32>
  ret <4 x i32> %shl
}

; CHECK: test_srad_3:
; CHECK: psrad   $32, %xmm0
; CHECK-NEXT: ret

; SSE Logical Shift Right

define <8 x i16> @test_srlw_1(<8 x i16> %InVec) {
entry:
  %shl = lshr <8 x i16> %InVec, <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  ret <8 x i16> %shl
}

; CHECK: test_srlw_1:
; CHECK: psrlw   $0, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_srlw_2(<8 x i16> %InVec) {
entry:
  %shl = lshr <8 x i16> %InVec, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  ret <8 x i16> %shl
}

; CHECK: test_srlw_2:
; CHECK: psrlw   $1, %xmm0
; CHECK-NEXT: ret

define <8 x i16> @test_srlw_3(<8 x i16> %InVec) {
entry:
  %shl = lshr <8 x i16> %InVec, <i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16>
  ret <8 x i16> %shl
}

; CHECK: test_srlw_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srld_1(<4 x i32> %InVec) {
entry:
  %shl = lshr <4 x i32> %InVec, <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %shl
}

; CHECK: test_srld_1:
; CHECK: psrld   $0, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srld_2(<4 x i32> %InVec) {
entry:
  %shl = lshr <4 x i32> %InVec, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %shl
}

; CHECK: test_srld_2:
; CHECK: psrld   $1, %xmm0
; CHECK-NEXT: ret

define <4 x i32> @test_srld_3(<4 x i32> %InVec) {
entry:
  %shl = lshr <4 x i32> %InVec, <i32 32, i32 32, i32 32, i32 32>
  ret <4 x i32> %shl
}

; CHECK: test_srld_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_srlq_1(<2 x i64> %InVec) {
entry:
  %shl = lshr <2 x i64> %InVec, <i64 0, i64 0>
  ret <2 x i64> %shl
}

; CHECK: test_srlq_1:
; CHECK: psrlq   $0, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_srlq_2(<2 x i64> %InVec) {
entry:
  %shl = lshr <2 x i64> %InVec, <i64 1, i64 1>
  ret <2 x i64> %shl
}

; CHECK: test_srlq_2:
; CHECK: psrlq   $1, %xmm0
; CHECK-NEXT: ret

define <2 x i64> @test_srlq_3(<2 x i64> %InVec) {
entry:
  %shl = lshr <2 x i64> %InVec, <i64 64, i64 64>
  ret <2 x i64> %shl
}

; CHECK: test_srlq_3:
; CHECK: xorps   %xmm0, %xmm0
; CHECK-NEXT: ret
