; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; Duplicates arm64'd vshift.ll

declare <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_urshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_urshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vrshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_srshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_srshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vrshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vrshldu(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vrshlds(<1 x i64>, <1 x i64>)

define <1 x i64> @test_urshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_urshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vrshldu(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: urshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_srshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_srshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vrshlds(<1 x i64> %lhs, <1 x i64> %rhs)
;CHECK: srshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}



