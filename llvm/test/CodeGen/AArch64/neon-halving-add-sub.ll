; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <8 x i8> @llvm.arm.neon.vhaddu.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vhadds.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_uhadd_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_uhadd_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vhaddu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: uhadd v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_shadd_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_shadd_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vhadds.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: shadd v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_uhadd_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uhadd_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vhaddu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: uhadd v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_shadd_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_shadd_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vhadds.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: shadd v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_uhadd_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uhadd_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vhaddu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: uhadd v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_shadd_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_shadd_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vhadds.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: shadd v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

declare <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_uhadd_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uhadd_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vhaddu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: uhadd v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_shadd_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_shadd_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vhadds.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: shadd v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

declare <2 x i32> @llvm.arm.neon.vhaddu.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vhadds.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_uhadd_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uhadd_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vhaddu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: uhadd v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_shadd_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_shadd_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vhadds.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: shadd v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vhaddu.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vhadds.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_uhadd_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uhadd_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vhaddu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: uhadd v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_shadd_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_shadd_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vhadds.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: shadd v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}


declare <8 x i8> @llvm.arm.neon.vhsubu.v8i8(<8 x i8>, <8 x i8>)
declare <8 x i8> @llvm.arm.neon.vhsubs.v8i8(<8 x i8>, <8 x i8>)

define <8 x i8> @test_uhsub_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_uhsub_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vhsubu.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: uhsub v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

define <8 x i8> @test_shsub_v8i8(<8 x i8> %lhs, <8 x i8> %rhs) {
; CHECK: test_shsub_v8i8:
  %tmp1 = call <8 x i8> @llvm.arm.neon.vhsubs.v8i8(<8 x i8> %lhs, <8 x i8> %rhs)
; CHECK: shsub v0.8b, v0.8b, v1.8b
  ret <8 x i8> %tmp1
}

declare <16 x i8> @llvm.arm.neon.vhsubu.v16i8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8>, <16 x i8>)

define <16 x i8> @test_uhsub_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_uhsub_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vhsubu.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: uhsub v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

define <16 x i8> @test_shsub_v16i8(<16 x i8> %lhs, <16 x i8> %rhs) {
; CHECK: test_shsub_v16i8:
  %tmp1 = call <16 x i8> @llvm.arm.neon.vhsubs.v16i8(<16 x i8> %lhs, <16 x i8> %rhs)
; CHECK: shsub v0.16b, v0.16b, v1.16b
  ret <16 x i8> %tmp1
}

declare <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16>, <4 x i16>)

define <4 x i16> @test_uhsub_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_uhsub_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vhsubu.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: uhsub v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

define <4 x i16> @test_shsub_v4i16(<4 x i16> %lhs, <4 x i16> %rhs) {
; CHECK: test_shsub_v4i16:
  %tmp1 = call <4 x i16> @llvm.arm.neon.vhsubs.v4i16(<4 x i16> %lhs, <4 x i16> %rhs)
; CHECK: shsub v0.4h, v0.4h, v1.4h
  ret <4 x i16> %tmp1
}

declare <8 x i16> @llvm.arm.neon.vhsubu.v8i16(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16>, <8 x i16>)

define <8 x i16> @test_uhsub_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_uhsub_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vhsubu.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: uhsub v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

define <8 x i16> @test_shsub_v8i16(<8 x i16> %lhs, <8 x i16> %rhs) {
; CHECK: test_shsub_v8i16:
  %tmp1 = call <8 x i16> @llvm.arm.neon.vhsubs.v8i16(<8 x i16> %lhs, <8 x i16> %rhs)
; CHECK: shsub v0.8h, v0.8h, v1.8h
  ret <8 x i16> %tmp1
}

declare <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32>, <2 x i32>)
declare <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32>, <2 x i32>)

define <2 x i32> @test_uhsub_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_uhsub_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vhsubu.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: uhsub v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

define <2 x i32> @test_shsub_v2i32(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK: test_shsub_v2i32:
  %tmp1 = call <2 x i32> @llvm.arm.neon.vhsubs.v2i32(<2 x i32> %lhs, <2 x i32> %rhs)
; CHECK: shsub v0.2s, v0.2s, v1.2s
  ret <2 x i32> %tmp1
}

declare <4 x i32> @llvm.arm.neon.vhsubu.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32>, <4 x i32>)

define <4 x i32> @test_uhsub_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_uhsub_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vhsubu.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: uhsub v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

define <4 x i32> @test_shsub_v4i32(<4 x i32> %lhs, <4 x i32> %rhs) {
; CHECK: test_shsub_v4i32:
  %tmp1 = call <4 x i32> @llvm.arm.neon.vhsubs.v4i32(<4 x i32> %lhs, <4 x i32> %rhs)
; CHECK: shsub v0.4s, v0.4s, v1.4s
  ret <4 x i32> %tmp1
}

