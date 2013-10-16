; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define i64 @test_vnegd_s64(i64 %a) {
; CHECK: test_vnegd_s64
; CHECK: neg {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vneg.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vneg1.i = tail call <1 x i64> @llvm.aarch64.neon.vneg(<1 x i64> %vneg.i)
  %0 = extractelement <1 x i64> %vneg1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.aarch64.neon.vneg(<1 x i64>)

define i8 @test_vqnegb_s8(i8 %a) {
; CHECK: test_vqnegb_s8
; CHECK: sqneg {{b[0-9]+}}, {{b[0-9]+}}
entry:
  %vqneg.i = insertelement <1 x i8> undef, i8 %a, i32 0
  %vqneg1.i = call <1 x i8> @llvm.arm.neon.vqneg.v1i8(<1 x i8> %vqneg.i)
  %0 = extractelement <1 x i8> %vqneg1.i, i32 0
  ret i8 %0
}

declare <1 x i8> @llvm.arm.neon.vqneg.v1i8(<1 x i8>)

define i16 @test_vqnegh_s16(i16 %a) {
; CHECK: test_vqnegh_s16
; CHECK: sqneg {{h[0-9]+}}, {{h[0-9]+}}
entry:
  %vqneg.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vqneg1.i = call <1 x i16> @llvm.arm.neon.vqneg.v1i16(<1 x i16> %vqneg.i)
  %0 = extractelement <1 x i16> %vqneg1.i, i32 0
  ret i16 %0
}

declare <1 x i16> @llvm.arm.neon.vqneg.v1i16(<1 x i16>)

define i32 @test_vqnegs_s32(i32 %a) {
; CHECK: test_vqnegs_s32
; CHECK: sqneg {{s[0-9]+}}, {{s[0-9]+}}
entry:
  %vqneg.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vqneg1.i = call <1 x i32> @llvm.arm.neon.vqneg.v1i32(<1 x i32> %vqneg.i)
  %0 = extractelement <1 x i32> %vqneg1.i, i32 0
  ret i32 %0
}

declare <1 x i32> @llvm.arm.neon.vqneg.v1i32(<1 x i32>)

define i64 @test_vqnegd_s64(i64 %a) {
; CHECK: test_vqnegd_s64
; CHECK: sqneg {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %vqneg.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vqneg1.i = call <1 x i64> @llvm.arm.neon.vqneg.v1i64(<1 x i64> %vqneg.i)
  %0 = extractelement <1 x i64> %vqneg1.i, i32 0
  ret i64 %0
}

declare <1 x i64> @llvm.arm.neon.vqneg.v1i64(<1 x i64>)