; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s
; intrinsic wrangling that arm64 does differently.

define i8 @test_vqmovunh_s16(i16 %a) {
; CHECK: test_vqmovunh_s16
; CHECK: sqxtun {{b[0-9]+}}, {{h[0-9]+}}
entry:
  %vqmovun.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vqmovun1.i = call <1 x i8> @llvm.arm.neon.vqmovnsu.v1i8(<1 x i16> %vqmovun.i)
  %0 = extractelement <1 x i8> %vqmovun1.i, i32 0
  ret i8 %0
}

define i16 @test_vqmovuns_s32(i32 %a) {
; CHECK: test_vqmovuns_s32
; CHECK: sqxtun {{h[0-9]+}}, {{s[0-9]+}}
entry:
  %vqmovun.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vqmovun1.i = call <1 x i16> @llvm.arm.neon.vqmovnsu.v1i16(<1 x i32> %vqmovun.i)
  %0 = extractelement <1 x i16> %vqmovun1.i, i32 0
  ret i16 %0
}

define i32 @test_vqmovund_s64(i64 %a) {
; CHECK: test_vqmovund_s64
; CHECK: sqxtun {{s[0-9]+}}, {{d[0-9]+}}
entry:
  %vqmovun.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vqmovun1.i = call <1 x i32> @llvm.arm.neon.vqmovnsu.v1i32(<1 x i64> %vqmovun.i)
  %0 = extractelement <1 x i32> %vqmovun1.i, i32 0
  ret i32 %0
}

declare <1 x i8> @llvm.arm.neon.vqmovnsu.v1i8(<1 x i16>)
declare <1 x i16> @llvm.arm.neon.vqmovnsu.v1i16(<1 x i32>)
declare <1 x i32> @llvm.arm.neon.vqmovnsu.v1i32(<1 x i64>)

define i8 @test_vqmovnh_s16(i16 %a) {
; CHECK: test_vqmovnh_s16
; CHECK: sqxtn {{b[0-9]+}}, {{h[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vqmovn1.i = call <1 x i8> @llvm.arm.neon.vqmovns.v1i8(<1 x i16> %vqmovn.i)
  %0 = extractelement <1 x i8> %vqmovn1.i, i32 0
  ret i8 %0
}

define i16 @test_vqmovns_s32(i32 %a) {
; CHECK: test_vqmovns_s32
; CHECK: sqxtn {{h[0-9]+}}, {{s[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vqmovn1.i = call <1 x i16> @llvm.arm.neon.vqmovns.v1i16(<1 x i32> %vqmovn.i)
  %0 = extractelement <1 x i16> %vqmovn1.i, i32 0
  ret i16 %0
}

define i32 @test_vqmovnd_s64(i64 %a) {
; CHECK: test_vqmovnd_s64
; CHECK: sqxtn {{s[0-9]+}}, {{d[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vqmovn1.i = call <1 x i32> @llvm.arm.neon.vqmovns.v1i32(<1 x i64> %vqmovn.i)
  %0 = extractelement <1 x i32> %vqmovn1.i, i32 0
  ret i32 %0
}

declare <1 x i8> @llvm.arm.neon.vqmovns.v1i8(<1 x i16>)
declare <1 x i16> @llvm.arm.neon.vqmovns.v1i16(<1 x i32>)
declare <1 x i32> @llvm.arm.neon.vqmovns.v1i32(<1 x i64>)

define i8 @test_vqmovnh_u16(i16 %a) {
; CHECK: test_vqmovnh_u16
; CHECK: uqxtn {{b[0-9]+}}, {{h[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i16> undef, i16 %a, i32 0
  %vqmovn1.i = call <1 x i8> @llvm.arm.neon.vqmovnu.v1i8(<1 x i16> %vqmovn.i)
  %0 = extractelement <1 x i8> %vqmovn1.i, i32 0
  ret i8 %0
}


define i16 @test_vqmovns_u32(i32 %a) {
; CHECK: test_vqmovns_u32
; CHECK: uqxtn {{h[0-9]+}}, {{s[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i32> undef, i32 %a, i32 0
  %vqmovn1.i = call <1 x i16> @llvm.arm.neon.vqmovnu.v1i16(<1 x i32> %vqmovn.i)
  %0 = extractelement <1 x i16> %vqmovn1.i, i32 0
  ret i16 %0
}

define i32 @test_vqmovnd_u64(i64 %a) {
; CHECK: test_vqmovnd_u64
; CHECK: uqxtn {{s[0-9]+}}, {{d[0-9]+}}
entry:
  %vqmovn.i = insertelement <1 x i64> undef, i64 %a, i32 0
  %vqmovn1.i = call <1 x i32> @llvm.arm.neon.vqmovnu.v1i32(<1 x i64> %vqmovn.i)
  %0 = extractelement <1 x i32> %vqmovn1.i, i32 0
  ret i32 %0
}

declare <1 x i8> @llvm.arm.neon.vqmovnu.v1i8(<1 x i16>)
declare <1 x i16> @llvm.arm.neon.vqmovnu.v1i16(<1 x i32>)
declare <1 x i32> @llvm.arm.neon.vqmovnu.v1i32(<1 x i64>)
