; RUN: llc -march=arm64 -aarch64-neon-syntax=apple < %s | FileCheck %s

define signext i8 @test_vmaxv_s8(<8 x i8> %a1) {
; CHECK: test_vmaxv_s8
; CHECK: smaxv.8b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v8i8(<8 x i8> %a1)
  %0 = trunc i32 %vmaxv.i to i8
  ret i8 %0
}

define signext i16 @test_vmaxv_s16(<4 x i16> %a1) {
; CHECK: test_vmaxv_s16
; CHECK: smaxv.4h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v4i16(<4 x i16> %a1)
  %0 = trunc i32 %vmaxv.i to i16
  ret i16 %0
}

define i32 @test_vmaxv_s32(<2 x i32> %a1) {
; CHECK: test_vmaxv_s32
; 2 x i32 is not supported by the ISA, thus, this is a special case
; CHECK: smaxp.2s v[[REGNUM:[0-9]+]], v0, v0
; CHECK-NEXT: fmov w0, s[[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v2i32(<2 x i32> %a1)
  ret i32 %vmaxv.i
}

define signext i8 @test_vmaxvq_s8(<16 x i8> %a1) {
; CHECK: test_vmaxvq_s8
; CHECK: smaxv.16b b[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.b w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8> %a1)
  %0 = trunc i32 %vmaxv.i to i8
  ret i8 %0
}

define signext i16 @test_vmaxvq_s16(<8 x i16> %a1) {
; CHECK: test_vmaxvq_s16
; CHECK: smaxv.8h h[[REGNUM:[0-9]+]], v0
; CHECK-NEXT: smov.h w0, v[[REGNUM]][0]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16> %a1)
  %0 = trunc i32 %vmaxv.i to i16
  ret i16 %0
}

define i32 @test_vmaxvq_s32(<4 x i32> %a1) {
; CHECK: test_vmaxvq_s32
; CHECK: smaxv.4s [[REGNUM:s[0-9]+]], v0
; CHECK-NEXT: fmov w0, [[REGNUM]]
; CHECK-NEXT: ret
entry:
  %vmaxv.i = tail call i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32> %a1)
  ret i32 %vmaxv.i
}

declare i32 @llvm.aarch64.neon.smaxv.i32.v4i32(<4 x i32>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v8i16(<8 x i16>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v16i8(<16 x i8>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v2i32(<2 x i32>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v4i16(<4 x i16>)
declare i32 @llvm.aarch64.neon.smaxv.i32.v8i8(<8 x i8>)

