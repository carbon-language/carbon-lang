; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqrshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqrshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqrshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqrshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}

  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqrshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqrshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vqrshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqrshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i8> @llvm.aarch64.neon.vqrshlu.v1i8(<1 x i8>, <1 x i8>)
declare <1 x i8> @llvm.aarch64.neon.vqrshls.v1i8(<1 x i8>, <1 x i8>)

define <1 x i8> @test_uqrshl_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_uqrshl_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqrshlu.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: uqrshl {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}

  ret <1 x i8> %tmp1
}

define <1 x i8> @test_sqrshl_v1i8_aarch64(<1 x i8> %lhs, <1 x i8> %rhs) {
; CHECK: test_sqrshl_v1i8_aarch64:
  %tmp1 = call <1 x i8> @llvm.aarch64.neon.vqrshls.v1i8(<1 x i8> %lhs, <1 x i8> %rhs)
;CHECK: sqrshl {{b[0-31]+}}, {{b[0-31]+}}, {{b[0-31]+}}
  ret <1 x i8> %tmp1
}

declare <1 x i16> @llvm.aarch64.neon.vqrshlu.v1i16(<1 x i16>, <1 x i16>)
declare <1 x i16> @llvm.aarch64.neon.vqrshls.v1i16(<1 x i16>, <1 x i16>)

define <1 x i16> @test_uqrshl_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_uqrshl_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqrshlu.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: uqrshl {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}

  ret <1 x i16> %tmp1
}

define <1 x i16> @test_sqrshl_v1i16_aarch64(<1 x i16> %lhs, <1 x i16> %rhs) {
; CHECK: test_sqrshl_v1i16_aarch64:
  %tmp1 = call <1 x i16> @llvm.aarch64.neon.vqrshls.v1i16(<1 x i16> %lhs, <1 x i16> %rhs)
;CHECK: sqrshl {{h[0-31]+}}, {{h[0-31]+}}, {{h[0-31]+}}
  ret <1 x i16> %tmp1
}

declare <1 x i32> @llvm.aarch64.neon.vqrshlu.v1i32(<1 x i32>, <1 x i32>)
declare <1 x i32> @llvm.aarch64.neon.vqrshls.v1i32(<1 x i32>, <1 x i32>)

define <1 x i32> @test_uqrshl_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_uqrshl_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqrshlu.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: uqrshl {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}

  ret <1 x i32> %tmp1
}

define <1 x i32> @test_sqrshl_v1i32_aarch64(<1 x i32> %lhs, <1 x i32> %rhs) {
; CHECK: test_sqrshl_v1i32_aarch64:
  %tmp1 = call <1 x i32> @llvm.aarch64.neon.vqrshls.v1i32(<1 x i32> %lhs, <1 x i32> %rhs)
;CHECK: sqrshl {{s[0-31]+}}, {{s[0-31]+}}, {{s[0-31]+}}
  ret <1 x i32> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vqrshlu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vqrshls.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_uqrshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_uqrshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqrshlu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: uqrshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}

  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sqrshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sqrshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vqrshls.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: sqrshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}



