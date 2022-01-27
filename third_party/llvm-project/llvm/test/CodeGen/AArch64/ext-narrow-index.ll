; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

; Tests of shufflevector where the index operand is half the width of the vector
; operands. We should get one ext instruction and not two.

; i8 tests
define <8 x i8> @i8_off0(<16 x i8> %arg1, <16 x i8> %arg2) {
; CHECK-LABEL: i8_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> %arg2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_off1(<16 x i8> %arg1, <16 x i8> %arg2) {
; CHECK-LABEL: i8_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #1
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> %arg2, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_off8(<16 x i8> %arg1, <16 x i8> %arg2) {
; CHECK-LABEL: i8_off8:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> %arg2, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_off15(<16 x i8> %arg1, <16 x i8> %arg2) {
; CHECK-LABEL: i8_off15:
; CHECK: ext v0.16b, v0.16b, v1.16b, #15
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> %arg2, <8 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_off22(<16 x i8> %arg1, <16 x i8> %arg2) {
; CHECK-LABEL: i8_off22:
; CHECK: ext v0.16b, v1.16b, v1.16b, #6
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> %arg2, <8 x i32> <i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29>
  ret <8 x i8> %shuffle
}

; i16 tests
define <4 x i16> @i16_off0(<8 x i16> %arg1, <8 x i16> %arg2) {
; CHECK-LABEL: i16_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> %arg2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_off1(<8 x i16> %arg1, <8 x i16> %arg2) {
; CHECK-LABEL: i16_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #2
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> %arg2, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_off7(<8 x i16> %arg1, <8 x i16> %arg2) {
; CHECK-LABEL: i16_off7:
; CHECK: ext v0.16b, v0.16b, v1.16b, #14
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> %arg2, <4 x i32> <i32 7, i32 8, i32 9, i32 10>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_off8(<8 x i16> %arg1, <8 x i16> %arg2) {
; CHECK-LABEL: i16_off8:
; CHECK: mov v0.16b, v1.16b
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> %arg2, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  ret <4 x i16> %shuffle
}

; i32 tests
define <2 x i32> @i32_off0(<4 x i32> %arg1, <4 x i32> %arg2) {
; CHECK-LABEL: i32_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> %arg2, <2 x i32> <i32 0, i32 1>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_off1(<4 x i32> %arg1, <4 x i32> %arg2) {
; CHECK-LABEL: i32_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #4
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> %arg2, <2 x i32> <i32 1, i32 2>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_off3(<4 x i32> %arg1, <4 x i32> %arg2) {
; CHECK-LABEL: i32_off3:
; CHECK: ext v0.16b, v0.16b, v1.16b, #12
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> %arg2, <2 x i32> <i32 3, i32 4>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_off4(<4 x i32> %arg1, <4 x i32> %arg2) {
; CHECK-LABEL: i32_off4:
; CHECK: mov v0.16b, v1.16b
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> %arg2, <2 x i32> <i32 4, i32 5>
  ret <2 x i32> %shuffle
}

; i64 tests
define <1 x i64> @i64_off0(<2 x i64> %arg1, <2 x i64> %arg2) {
; CHECK-LABEL: i64_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> %arg2, <1 x i32> <i32 0>
  ret <1 x i64> %shuffle
}

define <1 x i64> @i64_off1(<2 x i64> %arg1, <2 x i64> %arg2) {
; CHECK-LABEL: i64_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> %arg2, <1 x i32> <i32 1>
  ret <1 x i64> %shuffle
}

define <1 x i64> @i64_off2(<2 x i64> %arg1, <2 x i64> %arg2) {
; CHECK-LABEL: i64_off2:
; CHECK: mov v0.16b, v1.16b
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> %arg2, <1 x i32> <i32 2>
  ret <1 x i64> %shuffle
}

; i8 tests with second operand zero
define <8 x i8> @i8_zero_off0(<16 x i8> %arg1) {
; CHECK-LABEL: i8_zero_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_zero_off1(<16 x i8> %arg1) {
; CHECK-LABEL: i8_zero_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #1
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> zeroinitializer, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_zero_off8(<16 x i8> %arg1) {
; CHECK-LABEL: i8_zero_off8:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> zeroinitializer, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_zero_off15(<16 x i8> %arg1) {
; CHECK-LABEL: i8_zero_off15:
; CHECK: movi [[REG:v[0-9]+]].2d, #0
; CHECK: ext v0.16b, v0.16b, [[REG]].16b, #15
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> zeroinitializer, <8 x i32> <i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22>
  ret <8 x i8> %shuffle
}

define <8 x i8> @i8_zero_off22(<16 x i8> %arg1) {
; CHECK-LABEL: i8_zero_off22:
; CHECK: movi v0.2d, #0
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <16 x i8> %arg1, <16 x i8> zeroinitializer, <8 x i32> <i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29>
  ret <8 x i8> %shuffle
}

; i16 tests with second operand zero
define <4 x i16> @i16_zero_off0(<8 x i16> %arg1) {
; CHECK-LABEL: i16_zero_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_zero_off1(<8 x i16> %arg1) {
; CHECK-LABEL: i16_zero_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #2
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> zeroinitializer, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_zero_off7(<8 x i16> %arg1) {
; CHECK-LABEL: i16_zero_off7:
; CHECK: movi [[REG:v[0-9]+]].2d, #0
; CHECK: ext v0.16b, v0.16b, [[REG]].16b, #14
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> zeroinitializer, <4 x i32> <i32 7, i32 8, i32 9, i32 10>
  ret <4 x i16> %shuffle
}

define <4 x i16> @i16_zero_off8(<8 x i16> %arg1) {
; CHECK-LABEL: i16_zero_off8:
; CHECK: movi v0.2d, #0
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <8 x i16> %arg1, <8 x i16> zeroinitializer, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
  ret <4 x i16> %shuffle
}

; i32 tests with second operand zero
define <2 x i32> @i32_zero_off0(<4 x i32> %arg1) {
; CHECK-LABEL: i32_zero_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> zeroinitializer, <2 x i32> <i32 0, i32 1>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_zero_off1(<4 x i32> %arg1) {
; CHECK-LABEL: i32_zero_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #4
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> zeroinitializer, <2 x i32> <i32 1, i32 2>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_zero_off3(<4 x i32> %arg1) {
; CHECK-LABEL: i32_zero_off3:
; CHECK: movi [[REG:v[0-9]+]].2d, #0
; CHECK: ext v0.16b, v0.16b, [[REG]].16b, #12
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> zeroinitializer, <2 x i32> <i32 3, i32 4>
  ret <2 x i32> %shuffle
}

define <2 x i32> @i32_zero_off4(<4 x i32> %arg1) {
; CHECK-LABEL: i32_zero_off4:
; CHECK: movi v0.2d, #0
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <4 x i32> %arg1, <4 x i32> zeroinitializer, <2 x i32> <i32 4, i32 5>
  ret <2 x i32> %shuffle
}

; i64 tests with second operand zero
define <1 x i64> @i64_zero_off0(<2 x i64> %arg1) {
; CHECK-LABEL: i64_zero_off0:
; CHECK-NOT: mov
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> zeroinitializer, <1 x i32> <i32 0>
  ret <1 x i64> %shuffle
}

define <1 x i64> @i64_zero_off1(<2 x i64> %arg1) {
; CHECK-LABEL: i64_zero_off1:
; CHECK-NOT: mov
; CHECK: ext v0.16b, v0.16b, v0.16b, #8
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> zeroinitializer, <1 x i32> <i32 1>
  ret <1 x i64> %shuffle
}

define <1 x i64> @i64_zero_off2(<2 x i64> %arg1) {
; CHECK-LABEL: i64_zero_off2:
; CHECK: fmov d0, xzr
; CHECK-NOT: ext
; CHECK: ret
entry:
  %shuffle = shufflevector <2 x i64> %arg1, <2 x i64> zeroinitializer, <1 x i32> <i32 2>
  ret <1 x i64> %shuffle
}
