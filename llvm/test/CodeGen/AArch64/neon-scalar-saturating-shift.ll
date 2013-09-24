; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i8> @llvm.aarch64.neon.vqshlu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.aarch64.neon.vqshls.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqshl_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqshl_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqshlu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqshl {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqshl_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqshl_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqshls.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqshl {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i16> @llvm.aarch64.neon.vqshlu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.aarch64.neon.vqshls.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqshl_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqshl_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqshlu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqshl {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqshl_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqshl_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqshls.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqshl {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i32> @llvm.aarch64.neon.vqshlu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.aarch64.neon.vqshls.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqshl_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqshl_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqshlu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqshl {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

define <1 x i32> @test_sqshl_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqshl_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqshls.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqshl {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vqshlu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vqshls.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqshlu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqshls.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}


