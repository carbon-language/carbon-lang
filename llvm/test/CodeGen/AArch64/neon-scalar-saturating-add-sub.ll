; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqadd_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqadd_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqaddu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: uqadd d0, d0, d1
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqadd_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqadd_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqadds.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sqadd d0, d0, d1
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqsub_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqsub_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqsubu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: uqsub d0, d0, d1
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqsub_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqsub_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqsubs.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sqsub d0, d0, d1
  ret <1 x i64> %tmp1
}

declare <1 x i8> @llvm.aarch64.neon.vqaddu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.aarch64.neon.vqadds.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqadd_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqadd_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqaddu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqadd {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqadd_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqadd_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqadds.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqadd {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i8> @llvm.aarch64.neon.vqsubu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.aarch64.neon.vqsubs.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqsub_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqsub_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqsubu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqsub {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqsub_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqsub_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqsubs.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqsub {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i16> @llvm.aarch64.neon.vqaddu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.aarch64.neon.vqadds.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqadd_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqadd_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqaddu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqadd {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqadd_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqadd_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqadds.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqadd {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i16> @llvm.aarch64.neon.vqsubu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.aarch64.neon.vqsubs.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqsub_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqsub_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqsubu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqsub {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqsub_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqsub_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqsubs.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqsub {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i32> @llvm.aarch64.neon.vqaddu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.aarch64.neon.vqadds.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqadd_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqadd_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqaddu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqadd {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

define <1 x i32> @test_sqadd_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqadd_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqadds.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqadd {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i32> @llvm.aarch64.neon.vqsubu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.aarch64.neon.vqsubs.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqsub_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqsub_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqsubu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqsub {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

define <1 x i32> @test_sqsub_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqsub_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqsubs.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqsub {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vqaddu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vqadds.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqadd_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqadd_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqaddu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqadd {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqadd_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqadd_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqadds.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqadd {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vqsubu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vqsubs.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqsub_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqsub_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqsubu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqsub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqsub_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqsub_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqsubs.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqsub {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}
