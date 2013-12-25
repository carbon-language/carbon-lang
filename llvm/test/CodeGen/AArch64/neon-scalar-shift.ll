; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_ushl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_ushl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sshl_v1i64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sshl_v1i64:
  %tmp1 = call <1 x i64> @llvm.arm.neon.vshifts.v1i64(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

declare <1 x i64> @llvm.aarch64.neon.vshldu(<1 x i64>, <1 x i64>)
declare <1 x i64> @llvm.aarch64.neon.vshlds(<1 x i64>, <1 x i64>)

define <1 x i64> @test_ushl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_ushl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vshldu(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: ushl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_sshl_v1i64_aarch64(<1 x i64> %lhs, <1 x i64> %rhs) {
; CHECK: test_sshl_v1i64_aarch64:
  %tmp1 = call <1 x i64> @llvm.aarch64.neon.vshlds(<1 x i64> %lhs, <1 x i64> %rhs)
; CHECK: sshl {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret <1 x i64> %tmp1
}

define <1 x i64> @test_vtst_s64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: test_vtst_s64
; CHECK: cmtst {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = and <1 x i64> %a, %b
  %1 = icmp ne <1 x i64> %0, zeroinitializer
  %vtst.i = sext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vtst.i
}

define <1 x i64> @test_vtst_u64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: test_vtst_u64
; CHECK: cmtst {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
entry:
  %0 = and <1 x i64> %a, %b
  %1 = icmp ne <1 x i64> %0, zeroinitializer
  %vtst.i = sext <1 x i1> %1 to <1 x i64>
  ret <1 x i64> %vtst.i
}

define <1 x i64> @test_vsli_n_p64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: test_vsli_n_p64
; CHECK: sli {{d[0-9]+}}, {{d[0-9]+}}, #0
entry:
  %vsli_n2 = tail call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> %a, <1 x i64> %b, i32 0)
  ret <1 x i64> %vsli_n2
}

declare <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64>, <1 x i64>, i32)

define <2 x i64> @test_vsliq_n_p64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vsliq_n_p64
; CHECK: sli {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
entry:
  %vsli_n2 = tail call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> %a, <2 x i64> %b, i32 0)
  ret <2 x i64> %vsli_n2
}

declare <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64>, <2 x i64>, i32)

define <2 x i32> @test_vrsqrte_u32(<2 x i32> %a) {
; CHECK-LABEL: test_vrsqrte_u32
; CHECK: ursqrte {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
entry:
  %vrsqrte1.i = tail call <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32> %a)
  ret <2 x i32> %vrsqrte1.i
}

define <4 x i32> @test_vrsqrteq_u32(<4 x i32> %a) {
; CHECK-LABEL: test_vrsqrteq_u32
; CHECK: ursqrte {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
entry:
  %vrsqrte1.i = tail call <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32> %a)
  ret <4 x i32> %vrsqrte1.i
}

define <8 x i8> @test_vqshl_n_s8(<8 x i8> %a) {
; CHECK-LABEL: test_vqshl_n_s8
; CHECK: sqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
entry:
  %vqshl_n = tail call <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
  ret <8 x i8> %vqshl_n
}

declare <8 x i8> @llvm.arm.neon.vqshifts.v8i8(<8 x i8>, <8 x i8>)

define <16 x i8> @test_vqshlq_n_s8(<16 x i8> %a) {
; CHECK-LABEL: test_vqshlq_n_s8
; CHECK: sqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
entry:
  %vqshl_n = tail call <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
  ret <16 x i8> %vqshl_n
}

declare <16 x i8> @llvm.arm.neon.vqshifts.v16i8(<16 x i8>, <16 x i8>)

define <4 x i16> @test_vqshl_n_s16(<4 x i16> %a) {
; CHECK-LABEL: test_vqshl_n_s16
; CHECK: sqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
entry:
  %vqshl_n1 = tail call <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16> %a, <4 x i16> zeroinitializer)
  ret <4 x i16> %vqshl_n1
}

declare <4 x i16> @llvm.arm.neon.vqshifts.v4i16(<4 x i16>, <4 x i16>)

define <8 x i16> @test_vqshlq_n_s16(<8 x i16> %a) {
; CHECK-LABEL: test_vqshlq_n_s16
; CHECK: sqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
entry:
  %vqshl_n1 = tail call <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16> %a, <8 x i16> zeroinitializer)
  ret <8 x i16> %vqshl_n1
}

declare <8 x i16> @llvm.arm.neon.vqshifts.v8i16(<8 x i16>, <8 x i16>)

define <2 x i32> @test_vqshl_n_s32(<2 x i32> %a) {
; CHECK-LABEL: test_vqshl_n_s32
; CHECK: sqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
entry:
  %vqshl_n1 = tail call <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32> %a, <2 x i32> zeroinitializer)
  ret <2 x i32> %vqshl_n1
}

declare <2 x i32> @llvm.arm.neon.vqshifts.v2i32(<2 x i32>, <2 x i32>)

define <4 x i32> @test_vqshlq_n_s32(<4 x i32> %a) {
; CHECK-LABEL: test_vqshlq_n_s32
; CHECK: sqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
entry:
  %vqshl_n1 = tail call <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32> %a, <4 x i32> zeroinitializer)
  ret <4 x i32> %vqshl_n1
}

declare <4 x i32> @llvm.arm.neon.vqshifts.v4i32(<4 x i32>, <4 x i32>)

define <2 x i64> @test_vqshlq_n_s64(<2 x i64> %a) {
; CHECK-LABEL: test_vqshlq_n_s64
; CHECK: sqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
entry:
  %vqshl_n1 = tail call <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64> %a, <2 x i64> zeroinitializer)
  ret <2 x i64> %vqshl_n1
}

declare <2 x i64> @llvm.arm.neon.vqshifts.v2i64(<2 x i64>, <2 x i64>)

define <8 x i8> @test_vqshl_n_u8(<8 x i8> %a) {
; CHECK-LABEL: test_vqshl_n_u8
; CHECK: uqshl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
entry:
  %vqshl_n = tail call <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
  ret <8 x i8> %vqshl_n
}

declare <8 x i8> @llvm.arm.neon.vqshiftu.v8i8(<8 x i8>, <8 x i8>)

define <16 x i8> @test_vqshlq_n_u8(<16 x i8> %a) {
; CHECK-LABEL: test_vqshlq_n_u8
; CHECK: uqshl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
entry:
  %vqshl_n = tail call <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
  ret <16 x i8> %vqshl_n
}

declare <16 x i8> @llvm.arm.neon.vqshiftu.v16i8(<16 x i8>, <16 x i8>)

define <4 x i16> @test_vqshl_n_u16(<4 x i16> %a) {
; CHECK-LABEL: test_vqshl_n_u16
; CHECK: uqshl {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
entry:
  %vqshl_n1 = tail call <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16> %a, <4 x i16> zeroinitializer)
  ret <4 x i16> %vqshl_n1
}

declare <4 x i16> @llvm.arm.neon.vqshiftu.v4i16(<4 x i16>, <4 x i16>)

define <8 x i16> @test_vqshlq_n_u16(<8 x i16> %a) {
; CHECK-LABEL: test_vqshlq_n_u16
; CHECK: uqshl {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
entry:
  %vqshl_n1 = tail call <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16> %a, <8 x i16> zeroinitializer)
  ret <8 x i16> %vqshl_n1
}

declare <8 x i16> @llvm.arm.neon.vqshiftu.v8i16(<8 x i16>, <8 x i16>)

define <2 x i32> @test_vqshl_n_u32(<2 x i32> %a) {
; CHECK-LABEL: test_vqshl_n_u32
; CHECK: uqshl {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
entry:
  %vqshl_n1 = tail call <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32> %a, <2 x i32> zeroinitializer)
  ret <2 x i32> %vqshl_n1
}

declare <2 x i32> @llvm.arm.neon.vqshiftu.v2i32(<2 x i32>, <2 x i32>)

define <4 x i32> @test_vqshlq_n_u32(<4 x i32> %a) {
; CHECK-LABEL: test_vqshlq_n_u32
; CHECK: uqshl {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
entry:
  %vqshl_n1 = tail call <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32> %a, <4 x i32> zeroinitializer)
  ret <4 x i32> %vqshl_n1
}

declare <4 x i32> @llvm.arm.neon.vqshiftu.v4i32(<4 x i32>, <4 x i32>)

define <2 x i64> @test_vqshlq_n_u64(<2 x i64> %a) {
; CHECK-LABEL: test_vqshlq_n_u64
; CHECK: uqshl {{v[0-9]+}}.2d, {{v[0-9]+}}.2d,
entry:
  %vqshl_n1 = tail call <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64> %a, <2 x i64> zeroinitializer)
  ret <2 x i64> %vqshl_n1
}

declare <2 x i64> @llvm.arm.neon.vqshiftu.v2i64(<2 x i64>, <2 x i64>)

declare <4 x i32> @llvm.arm.neon.vrsqrte.v4i32(<4 x i32>)

declare <2 x i32> @llvm.arm.neon.vrsqrte.v2i32(<2 x i32>)
