; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_uqshl_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_uqshl_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: uqshl v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_sqshl_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_sqshl_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: sqshl v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_uqshl_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uqshl_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: uqshl v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_sqshl_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_sqshl_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: sqshl v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_uqshl_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uqshl_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: uqshl v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_sqshl_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_sqshl_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: sqshl v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

declare <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_uqshl_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uqshl_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: uqshl v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_sqshl_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_sqshl_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: sqshl v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

declare <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_uqshl_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uqshl_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: uqshl v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_sqshl_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_sqshl_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: sqshl v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_uqshl_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uqshl_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: uqshl v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_sqshl_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_sqshl_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: sqshl v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

declare <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: uqshl d0, d0, d1
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sqshl d0, d0, d1
  ret <1 x i64> %tmp1
}

declare <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64>, <2 x i64>)
declare <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64>, <2 x i64>)

define <2 x i64> @test_uqshl_v2i64(<2 x i64> %lhs, <2 x i64> %rhs) {
; CHECK: test_uqshl_v2i64:
  %tmp1 = call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %lhs, <2 x i64> %rhs)
; CHECK: uqshl v0.2d, v0.2d, v1.2d
  ret <2 x i64> %tmp1
}

define <2 x i64> @test_sqshl_v2i64(<2 x i64> %lhs, <2 x i64> %rhs) {
; CHECK: test_sqshl_v2i64:
  %tmp1 = call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %lhs, <2 x i64> %rhs)
; CHECK: sqshl v0.2d, v0.2d, v1.2d
  ret <2 x i64> %tmp1
}

