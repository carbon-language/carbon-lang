; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s

declare <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8>, <8 x i8>)

declare <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32>, <2 x i32>)

declare <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16>, <4 x i16>)

declare <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32>, <4 x i32>)

declare <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16>, <4 x i16>)

declare <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8>, <8 x i8>)

declare <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32>, <2 x i32>)

declare <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16>, <4 x i16>)

declare <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8>, <8 x i8>)

declare <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32>, <2 x i32>)

declare <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16>, <4 x i16>)

declare <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8>, <8 x i8>)

declare <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32>, <2 x i32>)

declare <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16>, <4 x i16>)

declare <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8>, <8 x i8>)

declare <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64>, <2 x i64>)

declare <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32>, <4 x i32>)

declare <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16>, <8 x i16>)

declare <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64>, <2 x i64>)

declare <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32>, <4 x i32>)

declare <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16>, <8 x i16>)

define <8 x i16> @test_vaddl_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vaddl_s8:
; CHECK: saddl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = sext <8 x i8> %a to <8 x i16>
  %vmovl.i2.i = sext <8 x i8> %b to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i, %vmovl.i2.i
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddl_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vaddl_s16:
; CHECK: saddl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = sext <4 x i16> %a to <4 x i32>
  %vmovl.i2.i = sext <4 x i16> %b to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i, %vmovl.i2.i
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddl_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vaddl_s32:
; CHECK: saddl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = sext <2 x i32> %a to <2 x i64>
  %vmovl.i2.i = sext <2 x i32> %b to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i, %vmovl.i2.i
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddl_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vaddl_u8:
; CHECK: uaddl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = zext <8 x i8> %a to <8 x i16>
  %vmovl.i2.i = zext <8 x i8> %b to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i, %vmovl.i2.i
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddl_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vaddl_u16:
; CHECK: uaddl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = zext <4 x i16> %a to <4 x i32>
  %vmovl.i2.i = zext <4 x i16> %b to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i, %vmovl.i2.i
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddl_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vaddl_u32:
; CHECK: uaddl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = zext <2 x i32> %a to <2 x i64>
  %vmovl.i2.i = zext <2 x i32> %b to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i, %vmovl.i2.i
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddl_high_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaddl_high_s8:
; CHECK: saddl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = sext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %shuffle.i.i2.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = sext <8 x i8> %shuffle.i.i2.i to <8 x i16>
  %add.i = add <8 x i16> %0, %1
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddl_high_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddl_high_s16:
; CHECK: saddl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = sext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %shuffle.i.i2.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = sext <4 x i16> %shuffle.i.i2.i to <4 x i32>
  %add.i = add <4 x i32> %0, %1
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddl_high_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddl_high_s32:
; CHECK: saddl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = sext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %shuffle.i.i2.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = sext <2 x i32> %shuffle.i.i2.i to <2 x i64>
  %add.i = add <2 x i64> %0, %1
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddl_high_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaddl_high_u8:
; CHECK: uaddl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = zext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %shuffle.i.i2.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = zext <8 x i8> %shuffle.i.i2.i to <8 x i16>
  %add.i = add <8 x i16> %0, %1
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddl_high_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddl_high_u16:
; CHECK: uaddl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = zext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %shuffle.i.i2.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = zext <4 x i16> %shuffle.i.i2.i to <4 x i32>
  %add.i = add <4 x i32> %0, %1
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddl_high_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddl_high_u32:
; CHECK: uaddl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = zext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %shuffle.i.i2.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = zext <2 x i32> %shuffle.i.i2.i to <2 x i64>
  %add.i = add <2 x i64> %0, %1
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddw_s8(<8 x i16> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vaddw_s8:
; CHECK: saddw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = sext <8 x i8> %b to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddw_s16(<4 x i32> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vaddw_s16:
; CHECK: saddw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = sext <4 x i16> %b to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddw_s32(<2 x i64> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vaddw_s32:
; CHECK: saddw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = sext <2 x i32> %b to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddw_u8(<8 x i16> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vaddw_u8:
; CHECK: uaddw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = zext <8 x i8> %b to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddw_u16(<4 x i32> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vaddw_u16:
; CHECK: uaddw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = zext <4 x i16> %b to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddw_u32(<2 x i64> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vaddw_u32:
; CHECK: uaddw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = zext <2 x i32> %b to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddw_high_s8(<8 x i16> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaddw_high_s8:
; CHECK: saddw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = sext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %add.i = add <8 x i16> %0, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddw_high_s16(<4 x i32> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddw_high_s16:
; CHECK: saddw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = sext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %add.i = add <4 x i32> %0, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddw_high_s32(<2 x i64> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddw_high_s32:
; CHECK: saddw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = sext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %add.i = add <2 x i64> %0, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vaddw_high_u8(<8 x i16> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vaddw_high_u8:
; CHECK: uaddw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = zext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %add.i = add <8 x i16> %0, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vaddw_high_u16(<4 x i32> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddw_high_u16:
; CHECK: uaddw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = zext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %add.i = add <4 x i32> %0, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vaddw_high_u32(<2 x i64> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddw_high_u32:
; CHECK: uaddw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = zext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %add.i = add <2 x i64> %0, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vsubl_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vsubl_s8:
; CHECK: ssubl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = sext <8 x i8> %a to <8 x i16>
  %vmovl.i2.i = sext <8 x i8> %b to <8 x i16>
  %sub.i = sub <8 x i16> %vmovl.i.i, %vmovl.i2.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubl_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vsubl_s16:
; CHECK: ssubl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = sext <4 x i16> %a to <4 x i32>
  %vmovl.i2.i = sext <4 x i16> %b to <4 x i32>
  %sub.i = sub <4 x i32> %vmovl.i.i, %vmovl.i2.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubl_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vsubl_s32:
; CHECK: ssubl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = sext <2 x i32> %a to <2 x i64>
  %vmovl.i2.i = sext <2 x i32> %b to <2 x i64>
  %sub.i = sub <2 x i64> %vmovl.i.i, %vmovl.i2.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubl_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vsubl_u8:
; CHECK: usubl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = zext <8 x i8> %a to <8 x i16>
  %vmovl.i2.i = zext <8 x i8> %b to <8 x i16>
  %sub.i = sub <8 x i16> %vmovl.i.i, %vmovl.i2.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubl_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vsubl_u16:
; CHECK: usubl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = zext <4 x i16> %a to <4 x i32>
  %vmovl.i2.i = zext <4 x i16> %b to <4 x i32>
  %sub.i = sub <4 x i32> %vmovl.i.i, %vmovl.i2.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubl_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vsubl_u32:
; CHECK: usubl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = zext <2 x i32> %a to <2 x i64>
  %vmovl.i2.i = zext <2 x i32> %b to <2 x i64>
  %sub.i = sub <2 x i64> %vmovl.i.i, %vmovl.i2.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubl_high_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsubl_high_s8:
; CHECK: ssubl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = sext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %shuffle.i.i2.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = sext <8 x i8> %shuffle.i.i2.i to <8 x i16>
  %sub.i = sub <8 x i16> %0, %1
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubl_high_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubl_high_s16:
; CHECK: ssubl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = sext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %shuffle.i.i2.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = sext <4 x i16> %shuffle.i.i2.i to <4 x i32>
  %sub.i = sub <4 x i32> %0, %1
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubl_high_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubl_high_s32:
; CHECK: ssubl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = sext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %shuffle.i.i2.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = sext <2 x i32> %shuffle.i.i2.i to <2 x i64>
  %sub.i = sub <2 x i64> %0, %1
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubl_high_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsubl_high_u8:
; CHECK: usubl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = zext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %shuffle.i.i2.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = zext <8 x i8> %shuffle.i.i2.i to <8 x i16>
  %sub.i = sub <8 x i16> %0, %1
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubl_high_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubl_high_u16:
; CHECK: usubl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = zext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %shuffle.i.i2.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = zext <4 x i16> %shuffle.i.i2.i to <4 x i32>
  %sub.i = sub <4 x i32> %0, %1
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubl_high_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubl_high_u32:
; CHECK: usubl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = zext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %shuffle.i.i2.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = zext <2 x i32> %shuffle.i.i2.i to <2 x i64>
  %sub.i = sub <2 x i64> %0, %1
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubw_s8(<8 x i16> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vsubw_s8:
; CHECK: ssubw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = sext <8 x i8> %b to <8 x i16>
  %sub.i = sub <8 x i16> %a, %vmovl.i.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubw_s16(<4 x i32> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vsubw_s16:
; CHECK: ssubw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = sext <4 x i16> %b to <4 x i32>
  %sub.i = sub <4 x i32> %a, %vmovl.i.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubw_s32(<2 x i64> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vsubw_s32:
; CHECK: ssubw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = sext <2 x i32> %b to <2 x i64>
  %sub.i = sub <2 x i64> %a, %vmovl.i.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubw_u8(<8 x i16> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vsubw_u8:
; CHECK: usubw {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8b
entry:
  %vmovl.i.i = zext <8 x i8> %b to <8 x i16>
  %sub.i = sub <8 x i16> %a, %vmovl.i.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubw_u16(<4 x i32> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vsubw_u16:
; CHECK: usubw {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4h
entry:
  %vmovl.i.i = zext <4 x i16> %b to <4 x i32>
  %sub.i = sub <4 x i32> %a, %vmovl.i.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubw_u32(<2 x i64> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vsubw_u32:
; CHECK: usubw {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2s
entry:
  %vmovl.i.i = zext <2 x i32> %b to <2 x i64>
  %sub.i = sub <2 x i64> %a, %vmovl.i.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubw_high_s8(<8 x i16> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsubw_high_s8:
; CHECK: ssubw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = sext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %sub.i = sub <8 x i16> %a, %0
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubw_high_s16(<4 x i32> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubw_high_s16:
; CHECK: ssubw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = sext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %sub.i = sub <4 x i32> %a, %0
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubw_high_s32(<2 x i64> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubw_high_s32:
; CHECK: ssubw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = sext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %sub.i = sub <2 x i64> %a, %0
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vsubw_high_u8(<8 x i16> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vsubw_high_u8:
; CHECK: usubw2 {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %0 = zext <8 x i8> %shuffle.i.i.i to <8 x i16>
  %sub.i = sub <8 x i16> %a, %0
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vsubw_high_u16(<4 x i32> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubw_high_u16:
; CHECK: usubw2 {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %0 = zext <4 x i16> %shuffle.i.i.i to <4 x i32>
  %sub.i = sub <4 x i32> %a, %0
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vsubw_high_u32(<2 x i64> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubw_high_u32:
; CHECK: usubw2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %0 = zext <2 x i32> %shuffle.i.i.i to <2 x i64>
  %sub.i = sub <2 x i64> %a, %0
  ret <2 x i64> %sub.i
}

define <8 x i8> @test_vaddhn_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddhn_s16:
; CHECK: addhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vaddhn.i = add <8 x i16> %a, %b
  %vaddhn1.i = lshr <8 x i16> %vaddhn.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vaddhn2.i = trunc <8 x i16> %vaddhn1.i to <8 x i8>
  ret <8 x i8> %vaddhn2.i
}

define <4 x i16> @test_vaddhn_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddhn_s32:
; CHECK: addhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vaddhn.i = add <4 x i32> %a, %b
  %vaddhn1.i = lshr <4 x i32> %vaddhn.i, <i32 16, i32 16, i32 16, i32 16>
  %vaddhn2.i = trunc <4 x i32> %vaddhn1.i to <4 x i16>
  ret <4 x i16> %vaddhn2.i
}

define <2 x i32> @test_vaddhn_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vaddhn_s64:
; CHECK: addhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vaddhn.i = add <2 x i64> %a, %b
  %vaddhn1.i = lshr <2 x i64> %vaddhn.i, <i64 32, i64 32>
  %vaddhn2.i = trunc <2 x i64> %vaddhn1.i to <2 x i32>
  ret <2 x i32> %vaddhn2.i
}

define <8 x i8> @test_vaddhn_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddhn_u16:
; CHECK: addhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vaddhn.i = add <8 x i16> %a, %b
  %vaddhn1.i = lshr <8 x i16> %vaddhn.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vaddhn2.i = trunc <8 x i16> %vaddhn1.i to <8 x i8>
  ret <8 x i8> %vaddhn2.i
}

define <4 x i16> @test_vaddhn_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddhn_u32:
; CHECK: addhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vaddhn.i = add <4 x i32> %a, %b
  %vaddhn1.i = lshr <4 x i32> %vaddhn.i, <i32 16, i32 16, i32 16, i32 16>
  %vaddhn2.i = trunc <4 x i32> %vaddhn1.i to <4 x i16>
  ret <4 x i16> %vaddhn2.i
}

define <2 x i32> @test_vaddhn_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vaddhn_u64:
; CHECK: addhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vaddhn.i = add <2 x i64> %a, %b
  %vaddhn1.i = lshr <2 x i64> %vaddhn.i, <i64 32, i64 32>
  %vaddhn2.i = trunc <2 x i64> %vaddhn1.i to <2 x i32>
  ret <2 x i32> %vaddhn2.i
}

define <16 x i8> @test_vaddhn_high_s16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddhn_high_s16:
; CHECK: addhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vaddhn.i.i = add <8 x i16> %a, %b
  %vaddhn1.i.i = lshr <8 x i16> %vaddhn.i.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vaddhn2.i.i = trunc <8 x i16> %vaddhn1.i.i to <8 x i8>
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vaddhn_high_s32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddhn_high_s32:
; CHECK: addhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vaddhn.i.i = add <4 x i32> %a, %b
  %vaddhn1.i.i = lshr <4 x i32> %vaddhn.i.i, <i32 16, i32 16, i32 16, i32 16>
  %vaddhn2.i.i = trunc <4 x i32> %vaddhn1.i.i to <4 x i16>
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vaddhn_high_s64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vaddhn_high_s64:
; CHECK: addhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vaddhn.i.i = add <2 x i64> %a, %b
  %vaddhn1.i.i = lshr <2 x i64> %vaddhn.i.i, <i64 32, i64 32>
  %vaddhn2.i.i = trunc <2 x i64> %vaddhn1.i.i to <2 x i32>
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <16 x i8> @test_vaddhn_high_u16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vaddhn_high_u16:
; CHECK: addhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vaddhn.i.i = add <8 x i16> %a, %b
  %vaddhn1.i.i = lshr <8 x i16> %vaddhn.i.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vaddhn2.i.i = trunc <8 x i16> %vaddhn1.i.i to <8 x i8>
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vaddhn_high_u32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vaddhn_high_u32:
; CHECK: addhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vaddhn.i.i = add <4 x i32> %a, %b
  %vaddhn1.i.i = lshr <4 x i32> %vaddhn.i.i, <i32 16, i32 16, i32 16, i32 16>
  %vaddhn2.i.i = trunc <4 x i32> %vaddhn1.i.i to <4 x i16>
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vaddhn_high_u64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vaddhn_high_u64:
; CHECK: addhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vaddhn.i.i = add <2 x i64> %a, %b
  %vaddhn1.i.i = lshr <2 x i64> %vaddhn.i.i, <i64 32, i64 32>
  %vaddhn2.i.i = trunc <2 x i64> %vaddhn1.i.i to <2 x i32>
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vaddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i8> @test_vraddhn_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vraddhn_s16:
; CHECK: raddhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vraddhn2.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i8> %vraddhn2.i
}

define <4 x i16> @test_vraddhn_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vraddhn_s32:
; CHECK: raddhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vraddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i16> %vraddhn2.i
}

define <2 x i32> @test_vraddhn_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vraddhn_s64:
; CHECK: raddhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vraddhn2.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i32> %vraddhn2.i
}

define <8 x i8> @test_vraddhn_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vraddhn_u16:
; CHECK: raddhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vraddhn2.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i8> %vraddhn2.i
}

define <4 x i16> @test_vraddhn_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vraddhn_u32:
; CHECK: raddhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vraddhn2.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i16> %vraddhn2.i
}

define <2 x i32> @test_vraddhn_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vraddhn_u64:
; CHECK: raddhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vraddhn2.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i32> %vraddhn2.i
}

define <16 x i8> @test_vraddhn_high_s16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vraddhn_high_s16:
; CHECK: raddhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vraddhn2.i.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vraddhn_high_s32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vraddhn_high_s32:
; CHECK: raddhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vraddhn2.i.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vraddhn_high_s64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vraddhn_high_s64:
; CHECK: raddhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vraddhn2.i.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <16 x i8> @test_vraddhn_high_u16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vraddhn_high_u16:
; CHECK: raddhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vraddhn2.i.i = tail call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vraddhn_high_u32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vraddhn_high_u32:
; CHECK: raddhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vraddhn2.i.i = tail call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vraddhn_high_u64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vraddhn_high_u64:
; CHECK: raddhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vraddhn2.i.i = tail call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vraddhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i8> @test_vsubhn_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubhn_s16:
; CHECK: subhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vsubhn.i = sub <8 x i16> %a, %b
  %vsubhn1.i = lshr <8 x i16> %vsubhn.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vsubhn2.i = trunc <8 x i16> %vsubhn1.i to <8 x i8>
  ret <8 x i8> %vsubhn2.i
}

define <4 x i16> @test_vsubhn_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubhn_s32:
; CHECK: subhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vsubhn.i = sub <4 x i32> %a, %b
  %vsubhn1.i = lshr <4 x i32> %vsubhn.i, <i32 16, i32 16, i32 16, i32 16>
  %vsubhn2.i = trunc <4 x i32> %vsubhn1.i to <4 x i16>
  ret <4 x i16> %vsubhn2.i
}

define <2 x i32> @test_vsubhn_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsubhn_s64:
; CHECK: subhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vsubhn.i = sub <2 x i64> %a, %b
  %vsubhn1.i = lshr <2 x i64> %vsubhn.i, <i64 32, i64 32>
  %vsubhn2.i = trunc <2 x i64> %vsubhn1.i to <2 x i32>
  ret <2 x i32> %vsubhn2.i
}

define <8 x i8> @test_vsubhn_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubhn_u16:
; CHECK: subhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vsubhn.i = sub <8 x i16> %a, %b
  %vsubhn1.i = lshr <8 x i16> %vsubhn.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vsubhn2.i = trunc <8 x i16> %vsubhn1.i to <8 x i8>
  ret <8 x i8> %vsubhn2.i
}

define <4 x i16> @test_vsubhn_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubhn_u32:
; CHECK: subhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vsubhn.i = sub <4 x i32> %a, %b
  %vsubhn1.i = lshr <4 x i32> %vsubhn.i, <i32 16, i32 16, i32 16, i32 16>
  %vsubhn2.i = trunc <4 x i32> %vsubhn1.i to <4 x i16>
  ret <4 x i16> %vsubhn2.i
}

define <2 x i32> @test_vsubhn_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsubhn_u64:
; CHECK: subhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vsubhn.i = sub <2 x i64> %a, %b
  %vsubhn1.i = lshr <2 x i64> %vsubhn.i, <i64 32, i64 32>
  %vsubhn2.i = trunc <2 x i64> %vsubhn1.i to <2 x i32>
  ret <2 x i32> %vsubhn2.i
}

define <16 x i8> @test_vsubhn_high_s16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubhn_high_s16:
; CHECK: subhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vsubhn.i.i = sub <8 x i16> %a, %b
  %vsubhn1.i.i = lshr <8 x i16> %vsubhn.i.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vsubhn2.i.i = trunc <8 x i16> %vsubhn1.i.i to <8 x i8>
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vsubhn_high_s32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubhn_high_s32:
; CHECK: subhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vsubhn.i.i = sub <4 x i32> %a, %b
  %vsubhn1.i.i = lshr <4 x i32> %vsubhn.i.i, <i32 16, i32 16, i32 16, i32 16>
  %vsubhn2.i.i = trunc <4 x i32> %vsubhn1.i.i to <4 x i16>
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vsubhn_high_s64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsubhn_high_s64:
; CHECK: subhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vsubhn.i.i = sub <2 x i64> %a, %b
  %vsubhn1.i.i = lshr <2 x i64> %vsubhn.i.i, <i64 32, i64 32>
  %vsubhn2.i.i = trunc <2 x i64> %vsubhn1.i.i to <2 x i32>
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <16 x i8> @test_vsubhn_high_u16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vsubhn_high_u16:
; CHECK: subhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vsubhn.i.i = sub <8 x i16> %a, %b
  %vsubhn1.i.i = lshr <8 x i16> %vsubhn.i.i, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vsubhn2.i.i = trunc <8 x i16> %vsubhn1.i.i to <8 x i8>
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vsubhn_high_u32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vsubhn_high_u32:
; CHECK: subhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vsubhn.i.i = sub <4 x i32> %a, %b
  %vsubhn1.i.i = lshr <4 x i32> %vsubhn.i.i, <i32 16, i32 16, i32 16, i32 16>
  %vsubhn2.i.i = trunc <4 x i32> %vsubhn1.i.i to <4 x i16>
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vsubhn_high_u64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsubhn_high_u64:
; CHECK: subhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vsubhn.i.i = sub <2 x i64> %a, %b
  %vsubhn1.i.i = lshr <2 x i64> %vsubhn.i.i, <i64 32, i64 32>
  %vsubhn2.i.i = trunc <2 x i64> %vsubhn1.i.i to <2 x i32>
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i8> @test_vrsubhn_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vrsubhn_s16:
; CHECK: rsubhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vrsubhn2.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i8> %vrsubhn2.i
}

define <4 x i16> @test_vrsubhn_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vrsubhn_s32:
; CHECK: rsubhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vrsubhn2.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i16> %vrsubhn2.i
}

define <2 x i32> @test_vrsubhn_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vrsubhn_s64:
; CHECK: rsubhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vrsubhn2.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i32> %vrsubhn2.i
}

define <8 x i8> @test_vrsubhn_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vrsubhn_u16:
; CHECK: rsubhn {{v[0-9]+}}.8b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vrsubhn2.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  ret <8 x i8> %vrsubhn2.i
}

define <4 x i16> @test_vrsubhn_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vrsubhn_u32:
; CHECK: rsubhn {{v[0-9]+}}.4h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vrsubhn2.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i16> %vrsubhn2.i
}

define <2 x i32> @test_vrsubhn_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vrsubhn_u64:
; CHECK: rsubhn {{v[0-9]+}}.2s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vrsubhn2.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i32> %vrsubhn2.i
}

define <16 x i8> @test_vrsubhn_high_s16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vrsubhn_high_s16:
; CHECK: rsubhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vrsubhn2.i.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vrsubhn_high_s32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vrsubhn_high_s32:
; CHECK: rsubhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vrsubhn2.i.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vrsubhn_high_s64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vrsubhn_high_s64:
; CHECK: rsubhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vrsubhn2.i.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <16 x i8> @test_vrsubhn_high_u16(<8 x i8> %r, <8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vrsubhn_high_u16:
; CHECK: rsubhn2 {{v[0-9]+}}.16b, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %vrsubhn2.i.i = tail call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
  %0 = bitcast <8 x i8> %r to <1 x i64>
  %1 = bitcast <8 x i8> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <16 x i8>
  ret <16 x i8> %2
}

define <8 x i16> @test_vrsubhn_high_u32(<4 x i16> %r, <4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vrsubhn_high_u32:
; CHECK: rsubhn2 {{v[0-9]+}}.8h, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vrsubhn2.i.i = tail call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
  %0 = bitcast <4 x i16> %r to <1 x i64>
  %1 = bitcast <4 x i16> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <8 x i16>
  ret <8 x i16> %2
}

define <4 x i32> @test_vrsubhn_high_u64(<2 x i32> %r, <2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vrsubhn_high_u64:
; CHECK: rsubhn2 {{v[0-9]+}}.4s, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %vrsubhn2.i.i = tail call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
  %0 = bitcast <2 x i32> %r to <1 x i64>
  %1 = bitcast <2 x i32> %vrsubhn2.i.i to <1 x i64>
  %shuffle.i.i = shufflevector <1 x i64> %0, <1 x i64> %1, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i.i to <4 x i32>
  ret <4 x i32> %2
}

define <8 x i16> @test_vabdl_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vabdl_s8:
; CHECK: sabdl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vabd.i.i = tail call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %vmovl.i.i = zext <8 x i8> %vabd.i.i to <8 x i16>
  ret <8 x i16> %vmovl.i.i
}

define <4 x i32> @test_vabdl_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vabdl_s16:
; CHECK: sabdl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vabd2.i.i = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %a, <4 x i16> %b)
  %vmovl.i.i = zext <4 x i16> %vabd2.i.i to <4 x i32>
  ret <4 x i32> %vmovl.i.i
}

define <2 x i64> @test_vabdl_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vabdl_s32:
; CHECK: sabdl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vabd2.i.i = tail call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %a, <2 x i32> %b)
  %vmovl.i.i = zext <2 x i32> %vabd2.i.i to <2 x i64>
  ret <2 x i64> %vmovl.i.i
}

define <8 x i16> @test_vabdl_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vabdl_u8:
; CHECK: uabdl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vabd.i.i = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a, <8 x i8> %b)
  %vmovl.i.i = zext <8 x i8> %vabd.i.i to <8 x i16>
  ret <8 x i16> %vmovl.i.i
}

define <4 x i32> @test_vabdl_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vabdl_u16:
; CHECK: uabdl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vabd2.i.i = tail call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %a, <4 x i16> %b)
  %vmovl.i.i = zext <4 x i16> %vabd2.i.i to <4 x i32>
  ret <4 x i32> %vmovl.i.i
}

define <2 x i64> @test_vabdl_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vabdl_u32:
; CHECK: uabdl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vabd2.i.i = tail call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %a, <2 x i32> %b)
  %vmovl.i.i = zext <2 x i32> %vabd2.i.i to <2 x i64>
  ret <2 x i64> %vmovl.i.i
}

define <8 x i16> @test_vabal_s8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vabal_s8:
; CHECK: sabal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vabd.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %b, <8 x i8> %c)
  %vmovl.i.i.i = zext <8 x i8> %vabd.i.i.i to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vabal_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vabal_s16:
; CHECK: sabal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vabd2.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %b, <4 x i16> %c)
  %vmovl.i.i.i = zext <4 x i16> %vabd2.i.i.i to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vabal_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vabal_s32:
; CHECK: sabal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vabd2.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %b, <2 x i32> %c)
  %vmovl.i.i.i = zext <2 x i32> %vabd2.i.i.i to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vabal_u8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vabal_u8:
; CHECK: uabal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vabd.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %b, <8 x i8> %c)
  %vmovl.i.i.i = zext <8 x i8> %vabd.i.i.i to <8 x i16>
  %add.i = add <8 x i16> %vmovl.i.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vabal_u16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vabal_u16:
; CHECK: uabal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vabd2.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %b, <4 x i16> %c)
  %vmovl.i.i.i = zext <4 x i16> %vabd2.i.i.i to <4 x i32>
  %add.i = add <4 x i32> %vmovl.i.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vabal_u32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vabal_u32:
; CHECK: uabal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vabd2.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %b, <2 x i32> %c)
  %vmovl.i.i.i = zext <2 x i32> %vabd2.i.i.i to <2 x i64>
  %add.i = add <2 x i64> %vmovl.i.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vabdl_high_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vabdl_high_s8:
; CHECK: sabdl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vabd.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %vmovl.i.i.i = zext <8 x i8> %vabd.i.i.i to <8 x i16>
  ret <8 x i16> %vmovl.i.i.i
}

define <4 x i32> @test_vabdl_high_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vabdl_high_s16:
; CHECK: sabdl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vabd2.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vmovl.i.i.i = zext <4 x i16> %vabd2.i.i.i to <4 x i32>
  ret <4 x i32> %vmovl.i.i.i
}

define <2 x i64> @test_vabdl_high_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vabdl_high_s32:
; CHECK: sabdl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vabd2.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vmovl.i.i.i = zext <2 x i32> %vabd2.i.i.i to <2 x i64>
  ret <2 x i64> %vmovl.i.i.i
}

define <8 x i16> @test_vabdl_high_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vabdl_high_u8:
; CHECK: uabdl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vabd.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %vmovl.i.i.i = zext <8 x i8> %vabd.i.i.i to <8 x i16>
  ret <8 x i16> %vmovl.i.i.i
}

define <4 x i32> @test_vabdl_high_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vabdl_high_u16:
; CHECK: uabdl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vabd2.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vmovl.i.i.i = zext <4 x i16> %vabd2.i.i.i to <4 x i32>
  ret <4 x i32> %vmovl.i.i.i
}

define <2 x i64> @test_vabdl_high_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vabdl_high_u32:
; CHECK: uabdl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vabd2.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vmovl.i.i.i = zext <2 x i32> %vabd2.i.i.i to <2 x i64>
  ret <2 x i64> %vmovl.i.i.i
}

define <8 x i16> @test_vabal_high_s8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vabal_high_s8:
; CHECK: sabal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vabd.i.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <8 x i8> %vabd.i.i.i.i to <8 x i16>
  %add.i.i = add <8 x i16> %vmovl.i.i.i.i, %a
  ret <8 x i16> %add.i.i
}

define <4 x i32> @test_vabal_high_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vabal_high_s16:
; CHECK: sabal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vabd2.i.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <4 x i16> %vabd2.i.i.i.i to <4 x i32>
  %add.i.i = add <4 x i32> %vmovl.i.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vabal_high_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vabal_high_s32:
; CHECK: sabal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vabd2.i.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <2 x i32> %vabd2.i.i.i.i to <2 x i64>
  %add.i.i = add <2 x i64> %vmovl.i.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <8 x i16> @test_vabal_high_u8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vabal_high_u8:
; CHECK: uabal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vabd.i.i.i.i = tail call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <8 x i8> %vabd.i.i.i.i to <8 x i16>
  %add.i.i = add <8 x i16> %vmovl.i.i.i.i, %a
  ret <8 x i16> %add.i.i
}

define <4 x i32> @test_vabal_high_u16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vabal_high_u16:
; CHECK: uabal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vabd2.i.i.i.i = tail call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <4 x i16> %vabd2.i.i.i.i to <4 x i32>
  %add.i.i = add <4 x i32> %vmovl.i.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vabal_high_u32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vabal_high_u32:
; CHECK: uabal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vabd2.i.i.i.i = tail call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vmovl.i.i.i.i = zext <2 x i32> %vabd2.i.i.i.i to <2 x i64>
  %add.i.i = add <2 x i64> %vmovl.i.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <8 x i16> @test_vmull_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vmull_s8:
; CHECK: smull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %a, <8 x i8> %b)
  ret <8 x i16> %vmull.i
}

define <4 x i32> @test_vmull_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vmull_s16:
; CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %a, <4 x i16> %b)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vmull_s32:
; CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %a, <2 x i32> %b)
  ret <2 x i64> %vmull2.i
}

define <8 x i16> @test_vmull_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vmull_u8:
; CHECK: umull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %a, <8 x i8> %b)
  ret <8 x i16> %vmull.i
}

define <4 x i32> @test_vmull_u16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vmull_u16:
; CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %a, <4 x i16> %b)
  ret <4 x i32> %vmull2.i
}

define <2 x i64> @test_vmull_u32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vmull_u32:
; CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %a, <2 x i32> %b)
  ret <2 x i64> %vmull2.i
}

define <8 x i16> @test_vmull_high_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmull_high_s8:
; CHECK: smull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  ret <8 x i16> %vmull.i.i
}

define <4 x i32> @test_vmull_high_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmull_high_s16:
; CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  ret <4 x i32> %vmull2.i.i
}

define <2 x i64> @test_vmull_high_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmull_high_s32:
; CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  ret <2 x i64> %vmull2.i.i
}

define <8 x i16> @test_vmull_high_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmull_high_u8:
; CHECK: umull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  ret <8 x i16> %vmull.i.i
}

define <4 x i32> @test_vmull_high_u16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vmull_high_u16:
; CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  ret <4 x i32> %vmull2.i.i
}

define <2 x i64> @test_vmull_high_u32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vmull_high_u32:
; CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  ret <2 x i64> %vmull2.i.i
}

define <8 x i16> @test_vmlal_s8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vmlal_s8:
; CHECK: smlal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
  %add.i = add <8 x i16> %vmull.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vmlal_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vmlal_s16:
; CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %add.i = add <4 x i32> %vmull2.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vmlal_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vmlal_s32:
; CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %add.i = add <2 x i64> %vmull2.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vmlal_u8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vmlal_u8:
; CHECK: umlal {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
  %add.i = add <8 x i16> %vmull.i.i, %a
  ret <8 x i16> %add.i
}

define <4 x i32> @test_vmlal_u16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vmlal_u16:
; CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %add.i = add <4 x i32> %vmull2.i.i, %a
  ret <4 x i32> %add.i
}

define <2 x i64> @test_vmlal_u32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vmlal_u32:
; CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %add.i = add <2 x i64> %vmull2.i.i, %a
  ret <2 x i64> %add.i
}

define <8 x i16> @test_vmlal_high_s8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmlal_high_s8:
; CHECK: smlal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %add.i.i = add <8 x i16> %vmull.i.i.i, %a
  ret <8 x i16> %add.i.i
}

define <4 x i32> @test_vmlal_high_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmlal_high_s16:
; CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmlal_high_s32:
; CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <8 x i16> @test_vmlal_high_u8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmlal_high_u8:
; CHECK: umlal2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %add.i.i = add <8 x i16> %vmull.i.i.i, %a
  ret <8 x i16> %add.i.i
}

define <4 x i32> @test_vmlal_high_u16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmlal_high_u16:
; CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_u32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmlal_high_u32:
; CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <8 x i16> @test_vmlsl_s8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vmlsl_s8:
; CHECK: smlsl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
  %sub.i = sub <8 x i16> %a, %vmull.i.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vmlsl_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vmlsl_s16:
; CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %sub.i = sub <4 x i32> %a, %vmull2.i.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vmlsl_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vmlsl_s32:
; CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %sub.i = sub <2 x i64> %a, %vmull2.i.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vmlsl_u8(<8 x i16> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK-LABEL: test_vmlsl_u8:
; CHECK: umlsl {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
  %sub.i = sub <8 x i16> %a, %vmull.i.i
  ret <8 x i16> %sub.i
}

define <4 x i32> @test_vmlsl_u16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vmlsl_u16:
; CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %sub.i = sub <4 x i32> %a, %vmull2.i.i
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vmlsl_u32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vmlsl_u32:
; CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %sub.i = sub <2 x i64> %a, %vmull2.i.i
  ret <2 x i64> %sub.i
}

define <8 x i16> @test_vmlsl_high_s8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmlsl_high_s8:
; CHECK: smlsl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i.i = tail call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %sub.i.i = sub <8 x i16> %a, %vmull.i.i.i
  ret <8 x i16> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmlsl_high_s16:
; CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = tail call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmlsl_high_s32:
; CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = tail call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <8 x i16> @test_vmlsl_high_u8(<8 x i16> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK-LABEL: test_vmlsl_high_u8:
; CHECK: umlsl2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %c, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i.i = tail call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  %sub.i.i = sub <8 x i16> %a, %vmull.i.i.i
  ret <8 x i16> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_u16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vmlsl_high_u16:
; CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = tail call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_u32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vmlsl_high_u32:
; CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = tail call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <4 x i32> @test_vqdmull_s16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: test_vqdmull_s16:
; CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vqdmull2.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> %b)
  ret <4 x i32> %vqdmull2.i
}

define <2 x i64> @test_vqdmull_s32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: test_vqdmull_s32:
; CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vqdmull2.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> %b)
  ret <2 x i64> %vqdmull2.i
}

define <4 x i32> @test_vqdmlal_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vqdmlal_s16:
; CHECK: sqdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vqdmlal2.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %vqdmlal4.i = tail call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i)
  ret <4 x i32> %vqdmlal4.i
}

define <2 x i64> @test_vqdmlal_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vqdmlal_s32:
; CHECK: sqdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vqdmlal2.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %vqdmlal4.i = tail call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i)
  ret <2 x i64> %vqdmlal4.i
}

define <4 x i32> @test_vqdmlsl_s16(<4 x i32> %a, <4 x i16> %b, <4 x i16> %c) {
; CHECK-LABEL: test_vqdmlsl_s16:
; CHECK: sqdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
entry:
  %vqdmlsl2.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %b, <4 x i16> %c)
  %vqdmlsl4.i = tail call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i)
  ret <4 x i32> %vqdmlsl4.i
}

define <2 x i64> @test_vqdmlsl_s32(<2 x i64> %a, <2 x i32> %b, <2 x i32> %c) {
; CHECK-LABEL: test_vqdmlsl_s32:
; CHECK: sqdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vqdmlsl2.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %b, <2 x i32> %c)
  %vqdmlsl4.i = tail call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i)
  ret <2 x i64> %vqdmlsl4.i
}

define <4 x i32> @test_vqdmull_high_s16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: test_vqdmull_high_s16:
; CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmull2.i.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  ret <4 x i32> %vqdmull2.i.i
}

define <2 x i64> @test_vqdmull_high_s32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: test_vqdmull_high_s32:
; CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmull2.i.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  ret <2 x i64> %vqdmull2.i.i
}

define <4 x i32> @test_vqdmlal_high_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vqdmlal_high_s16:
; CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmlal2.i.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vqdmlal4.i.i = tail call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal2.i.i)
  ret <4 x i32> %vqdmlal4.i.i
}

define <2 x i64> @test_vqdmlal_high_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vqdmlal_high_s32:
; CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmlal2.i.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vqdmlal4.i.i = tail call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal2.i.i)
  ret <2 x i64> %vqdmlal4.i.i
}

define <4 x i32> @test_vqdmlsl_high_s16(<4 x i32> %a, <8 x i16> %b, <8 x i16> %c) {
; CHECK-LABEL: test_vqdmlsl_high_s16:
; CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %shuffle.i3.i = shufflevector <8 x i16> %c, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmlsl2.i.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %shuffle.i3.i)
  %vqdmlsl4.i.i = tail call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl2.i.i)
  ret <4 x i32> %vqdmlsl4.i.i
}

define <2 x i64> @test_vqdmlsl_high_s32(<2 x i64> %a, <4 x i32> %b, <4 x i32> %c) {
; CHECK-LABEL: test_vqdmlsl_high_s32:
; CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %shuffle.i3.i = shufflevector <4 x i32> %c, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmlsl2.i.i = tail call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %shuffle.i3.i)
  %vqdmlsl4.i.i = tail call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl2.i.i)
  ret <2 x i64> %vqdmlsl4.i.i
}

define <8 x i16> @test_vmull_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: test_vmull_p8:
; CHECK: pmull {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
entry:
  %vmull.i = tail call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> %a, <8 x i8> %b)
  ret <8 x i16> %vmull.i
}

define <8 x i16> @test_vmull_high_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: test_vmull_high_p8:
; CHECK: pmull2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
entry:
  %shuffle.i.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %shuffle.i3.i = shufflevector <16 x i8> %b, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vmull.i.i = tail call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> %shuffle.i.i, <8 x i8> %shuffle.i3.i)
  ret <8 x i16> %vmull.i.i
}

define i128 @test_vmull_p64(i64 %a, i64 %b) #4 {
; CHECK-LABEL: test_vmull_p64
; CHECK: pmull {{v[0-9]+}}.1q, {{v[0-9]+}}.1d, {{v[0-9]+}}.1d
entry:
  %vmull2.i = tail call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %a, i64 %b)
  %vmull3.i = bitcast <16 x i8> %vmull2.i to i128
  ret i128 %vmull3.i
}

define i128 @test_vmull_high_p64(<2 x i64> %a, <2 x i64> %b) #4 {
; CHECK-LABEL: test_vmull_high_p64
; CHECK: pmull2 {{v[0-9]+}}.1q, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
entry:
  %0 = extractelement <2 x i64> %a, i32 1
  %1 = extractelement <2 x i64> %b, i32 1
  %vmull2.i.i = tail call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %0, i64 %1) #1
  %vmull3.i.i = bitcast <16 x i8> %vmull2.i.i to i128
  ret i128 %vmull3.i.i
}

declare <16 x i8> @llvm.aarch64.neon.pmull64(i64, i64) #5


