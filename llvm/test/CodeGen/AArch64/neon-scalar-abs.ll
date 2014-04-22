; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s
; arm64 has tests for i64 versions, uses different approach for others.

define i64 @test_vabsd_s64(i64 %a) {
; CHECK: test_vabsd_s64
; CHECK: abs {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vabs.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vabs1.i = tail call <1 x i64> @llvm.aarch64.neon.vabs(<1 x i64> %vabs.i)
  %0 = extractelement <1 x i64> %vabs1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vabs(<1 x i64>)

define i8 @test_vqabsb_s8(i8 %a) {
; CHECK: test_vqabsb_s8
; CHECK: sqabs {{b[0-9]+}}, {{b[0-9]+}}
entry:
  %vqabs.i = insertelement <1 x i8> undef, i8 %a, i32 0
  %vqabs1.i = call <1 x i8> @llvm.arm.neon.vqabs.v1i8(<1 x i8> %vqabs.i)
  %0 = extractelement <1 x i8> %vqabs1.i, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.arm.neon.vqabs.v1i8(<1 x i8>)

define i16 @test_vqabsh_s16(i16 %a) {
; CHECK: test_vqabsh_s16
; CHECK: sqabs {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %vqabs.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vqabs1.i = call <1 x i16> @llvm.arm.neon.vqabs.v1i16(<1 x i16> %vqabs.i)
  %0 = extractelement <1 x i16> %vqabs1.i, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.arm.neon.vqabs.v1i16(<1 x i16>)

define i32 @test_vqabss_s32(i32 %a) {
; CHECK: test_vqabss_s32
; CHECK: sqabs {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vqabs.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vqabs1.i = call <1 x i32> @llvm.arm.neon.vqabs.v1i32(<1 x i32> %vqabs.i)
  %0 = extractelement <1 x i32> %vqabs1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.arm.neon.vqabs.v1i32(<1 x i32>)

define i64 @test_vqabsd_s64(i64 %a) {
; CHECK: test_vqabsd_s64
; CHECK: sqabs {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vqabs.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vqabs1.i = call <1 x i64> @llvm.arm.neon.vqabs.v1i64(<1 x i64> %vqabs.i)
  %0 = extractelement <1 x i64> %vqabs1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.arm.neon.vqabs.v1i64(<1 x i64>)
