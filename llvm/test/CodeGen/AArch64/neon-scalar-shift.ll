; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_ushl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_ushl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: ushl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}

  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vshldu(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vshlds(<1 x i64>, <1 x i64>)

define <1 x i64> @test_ushl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_ushl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vshldu(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: ushl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vshlds(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sshl {{d[0-31]+}}, {{d[0-31]+}}, {{d[0-31]+}}
  ret <1 x i64> %tmp1
}


