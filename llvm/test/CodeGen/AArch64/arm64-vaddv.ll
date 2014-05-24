; RUN: llc -march=arm64 -aarch64-neon-syntax=apple < %s -mcpu=cyclone | FileCheck %s

define signext i8 @test_vaddv_s8(<8 x i8> %a1) {
; CHECK-LABEL: test_vaddv_s8:
; CHECK: addv.8b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8> %a1)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define signext i16 @test_vaddv_s16(<4 x i16> %a1) {
; CHECK-LABEL: test_vaddv_s16:
; CHECK: addv.4h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16> %a1)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddv_s32(<2 x i32> %a1) {
; CHECK-LABEL: test_vaddv_s32:
; 2 x i32 is not supported by the ISA, thus, this is a special case
; CHECK: addp.2s v[[REGNUM:[0-9]+]], v0, v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32> %a1)
  ret i32 %vaddv.i
}

define i64 @test_vaddv_s64(<2 x i64> %a1) {
; CHECK-LABEL: test_vaddv_s64:
; CHECK: addp.2d [[REGNUM:d[0-9]+]], v0
; CHECK-NEXT: fmov x0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64> %a1)
  ret i64 %vaddv.i
}

define zeroext i8 @test_vaddv_u8(<8 x i8> %a1) {
; CHECK-LABEL: test_vaddv_u8:
; CHECK: addv.8b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v8i8(<8 x i8> %a1)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define i32 @test_vaddv_u8_masked(<8 x i8> %a1) {
; CHECK-LABEL: test_vaddv_u8_masked:
; CHECK: addv.8b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v8i8(<8 x i8> %a1)
  %0 = and i32 %vaddv.i, 511 ; 0x1ff
  ret i32 %0
}

define zeroext i16 @test_vaddv_u16(<4 x i16> %a1) {
; CHECK-LABEL: test_vaddv_u16:
; CHECK: addv.4h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v4i16(<4 x i16> %a1)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddv_u16_masked(<4 x i16> %a1) {
; CHECK-LABEL: test_vaddv_u16_masked:
; CHECK: addv.4h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v4i16(<4 x i16> %a1)
  %0 = and i32 %vaddv.i, 3276799 ; 0x31ffff
  ret i32 %0
}

define i32 @test_vaddv_u32(<2 x i32> %a1) {
; CHECK-LABEL: test_vaddv_u32:
; 2 x i32 is not supported by the ISA, thus, this is a special case
; CHECK: addp.2s v[[REGNUM:[0-9]+]], v0, v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v2i32(<2 x i32> %a1)
  ret i32 %vaddv.i
}

define float @test_vaddv_f32(<2 x float> %a1) {
; CHECK-LABEL: test_vaddv_f32:
; CHECK: faddp.2s s0, v0
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call float @llvm.aarch64.neon.faddv.f32.v2f32(<2 x float> %a1)
  ret float %vaddv.i
}

define float @test_vaddv_v4f32(<4 x float> %a1) {
; CHECK-LABEL: test_vaddv_v4f32:
; CHECK: faddp.4s [[REGNUM:v[0-9]+]], v0, v0
; CHECK: faddp.2s s0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float> %a1)
  ret float %vaddv.i
}

define double @test_vaddv_f64(<2 x double> %a1) {
; CHECK-LABEL: test_vaddv_f64:
; CHECK: faddp.2d d0, v0
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double> %a1)
  ret double %vaddv.i
}

define i64 @test_vaddv_u64(<2 x i64> %a1) {
; CHECK-LABEL: test_vaddv_u64:
; CHECK: addp.2d [[REGNUM:d[0-9]+]], v0
; CHECK-NEXT: fmov x0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a1)
  ret i64 %vaddv.i
}

define <1 x i64> @test_vaddv_u64_to_vec(<2 x i64> %a1) {
; CHECK-LABEL: test_vaddv_u64_to_vec:
; CHECK: addp.2d d0, v0
; CHECK-NOT: fmov
; CHECK-NOT: ins
; CHECK: ret
entry:
  %vaddv.i = tail call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a1)
  %vec = insertelement <1 x i64> undef, i64 %vaddv.i, i32 0
  ret <1 x i64> %vec
}

define signext i8 @test_vaddvq_s8(<16 x i8> %a1) {
; CHECK-LABEL: test_vaddvq_s8:
; CHECK: addv.16b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8> %a1)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define signext i16 @test_vaddvq_s16(<8 x i16> %a1) {
; CHECK-LABEL: test_vaddvq_s16:
; CHECK: addv.8h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16> %a1)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddvq_s32(<4 x i32> %a1) {
; CHECK-LABEL: test_vaddvq_s32:
; CHECK: addv.4s [[REGNUM:s[0-9]+]], v0
; CHECK-NEXT: fmov w0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32> %a1)
  ret i32 %vaddv.i
}

define zeroext i8 @test_vaddvq_u8(<16 x i8> %a1) {
; CHECK-LABEL: test_vaddvq_u8:
; CHECK: addv.16b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v16i8(<16 x i8> %a1)
  %0 = trunc i32 %vaddv.i to i8
  ret i8 %0
}

define zeroext i16 @test_vaddvq_u16(<8 x i16> %a1) {
; CHECK-LABEL: test_vaddvq_u16:
; CHECK: addv.8h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v8i16(<8 x i16> %a1)
  %0 = trunc i32 %vaddv.i to i16
  ret i16 %0
}

define i32 @test_vaddvq_u32(<4 x i32> %a1) {
; CHECK-LABEL: test_vaddvq_u32:
; CHECK: addv.4s [[REGNUM:s[0-9]+]], v0
; CHECK-NEXT: fmov [[FMOVRES:w[0-9]+]], [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vaddv.i = tail call i32 @llvm.aarch64.neon.uaddv.i32.v4i32(<4 x i32> %a1)
  ret i32 %vaddv.i
}

declare i32 @llvm.aarch64.neon.uaddv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.uaddv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.uaddv.i32.v16i8(<16 x i8>)

declare i32 @llvm.aarch64.neon.saddv.i32.v4i32(<4 x i32>)

declare i32 @llvm.aarch64.neon.saddv.i32.v8i16(<8 x i16>)

declare i32 @llvm.aarch64.neon.saddv.i32.v16i8(<16 x i8>)

declare i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64>)

declare i32 @llvm.aarch64.neon.uaddv.i32.v2i32(<2 x i32>)

declare i32 @llvm.aarch64.neon.uaddv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.uaddv.i32.v8i8(<8 x i8>)

declare i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32>)

declare i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64>)

declare i32 @llvm.aarch64.neon.saddv.i32.v4i16(<4 x i16>)

declare i32 @llvm.aarch64.neon.saddv.i32.v8i8(<8 x i8>)

declare float @llvm.aarch64.neon.faddv.f32.v2f32(<2 x float> %a1)
declare float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float> %a1)
declare double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double> %a1)
