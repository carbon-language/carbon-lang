; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s
; This test is just intrinsic pumping. arm64 has its own tbl/tbx tests.

declare <16 x i8> @llvm.aarch64.neon.vtbx4.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbx3.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbx2.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbx1.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbx4.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbx3.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8>, <16 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8>, <8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8>, <8 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbl4.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbl3.v16i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbl2.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)

declare <16 x i8> @llvm.aarch64.neon.vtbl1.v16i8(<16 x i8>, <16 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbl4.v8i8(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>)

declare <8 x i8> @llvm.aarch64.neon.vtbl3.v8i8(<16 x i8>, <16 x i8>, <16 x i8>, <8 x i8>)

define <8 x i8> @test_vtbl1_s8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtbl1_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl11.i
}

define <8 x i8> @test_vqtbl1_s8(<16 x i8> %a, <8 x i8> %b) {
; CHECK: test_vqtbl1_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %vtbl1.i
}

define <8 x i8> @test_vtbl2_s8([2 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl2_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 1
  %vtbl1.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl17.i
}

define <8 x i8> @test_vqtbl2_s8([2 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl2_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl2.i
}

define <8 x i8> @test_vtbl3_s8([3 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl3_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %b)
  ret <8 x i8> %vtbl212.i
}

define <8 x i8> @test_vqtbl3_s8([3 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl3_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl3.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl3.i
}

define <8 x i8> @test_vtbl4_s8([4 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl4_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 3
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl215.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %__a.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl215.i, <8 x i8> %b)
  ret <8 x i8> %vtbl216.i
}

define <8 x i8> @test_vqtbl4_s8([4 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl4_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl4.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl4.i
}

define <16 x i8> @test_vqtbl1q_s8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vqtbl1q_s8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbl1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl1.v16i8(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %vtbl1.i
}

define <16 x i8> @test_vqtbl2q_s8([2 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl2q_s8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl2.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl2.i
}

define <16 x i8> @test_vqtbl3q_s8([3 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl3q_s8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl3.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl3.i
}

define <16 x i8> @test_vqtbl4q_s8([4 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl4q_s8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl4.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl4.i
}

define <8 x i8> @test_vtbx1_s8(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK: test_vtbx1_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl11.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx2_s8(<8 x i8> %a, [2 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx2_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 1
  %vtbx1.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %vtbx1.i, <8 x i8> %c)
  ret <8 x i8> %vtbx17.i
}

define <8 x i8> @test_vtbx3_s8(<8 x i8> %a, [3 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx3_s8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl212.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx4_s8(<8 x i8> %a, [4 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx4_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 3
  %vtbx2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx215.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %__b.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %vtbx2.i, <16 x i8> %vtbx215.i, <8 x i8> %c)
  ret <8 x i8> %vtbx216.i
}

define <8 x i8> @test_vqtbx1_s8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) {
; CHECK: test_vqtbx1_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbx1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c)
  ret <8 x i8> %vtbx1.i
}

define <8 x i8> @test_vqtbx2_s8(<8 x i8> %a, [2 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx2_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx2.i
}

define <8 x i8> @test_vqtbx3_s8(<8 x i8> %a, [3 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx3_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx3.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx3.i
}

define <8 x i8> @test_vqtbx4_s8(<8 x i8> %a, [4 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx4_s8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx4.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx4.i
}

define <16 x i8> @test_vqtbx1q_s8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK: test_vqtbx1q_s8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbx1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c)
  ret <16 x i8> %vtbx1.i
}

define <16 x i8> @test_vqtbx2q_s8(<16 x i8> %a, [2 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx2q_s8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx2.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx2.i
}

define <16 x i8> @test_vqtbx3q_s8(<16 x i8> %a, [3 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx3q_s8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx3.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx3.i
}

define <16 x i8> @test_vqtbx4q_s8(<16 x i8> %a, [4 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx4q_s8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx4.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx4.i
}

define <8 x i8> @test_vtbl1_u8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtbl1_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl11.i
}

define <8 x i8> @test_vqtbl1_u8(<16 x i8> %a, <8 x i8> %b) {
; CHECK: test_vqtbl1_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %vtbl1.i
}

define <8 x i8> @test_vtbl2_u8([2 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl2_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 1
  %vtbl1.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl17.i
}

define <8 x i8> @test_vqtbl2_u8([2 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl2_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl2.i
}

define <8 x i8> @test_vtbl3_u8([3 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl3_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %b)
  ret <8 x i8> %vtbl212.i
}

define <8 x i8> @test_vqtbl3_u8([3 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl3_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl3.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl3.i
}

define <8 x i8> @test_vtbl4_u8([4 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl4_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 3
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl215.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %__a.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl215.i, <8 x i8> %b)
  ret <8 x i8> %vtbl216.i
}

define <8 x i8> @test_vqtbl4_u8([4 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl4_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl4.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl4.i
}

define <16 x i8> @test_vqtbl1q_u8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vqtbl1q_u8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbl1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl1.v16i8(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %vtbl1.i
}

define <16 x i8> @test_vqtbl2q_u8([2 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl2q_u8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl2.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl2.i
}

define <16 x i8> @test_vqtbl3q_u8([3 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl3q_u8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl3.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl3.i
}

define <16 x i8> @test_vqtbl4q_u8([4 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl4q_u8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl4.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl4.i
}

define <8 x i8> @test_vtbx1_u8(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK: test_vtbx1_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl11.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx2_u8(<8 x i8> %a, [2 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx2_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 1
  %vtbx1.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %vtbx1.i, <8 x i8> %c)
  ret <8 x i8> %vtbx17.i
}

define <8 x i8> @test_vtbx3_u8(<8 x i8> %a, [3 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx3_u8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl212.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx4_u8(<8 x i8> %a, [4 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx4_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 3
  %vtbx2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx215.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %__b.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %vtbx2.i, <16 x i8> %vtbx215.i, <8 x i8> %c)
  ret <8 x i8> %vtbx216.i
}

define <8 x i8> @test_vqtbx1_u8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) {
; CHECK: test_vqtbx1_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbx1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c)
  ret <8 x i8> %vtbx1.i
}

define <8 x i8> @test_vqtbx2_u8(<8 x i8> %a, [2 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx2_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx2.i
}

define <8 x i8> @test_vqtbx3_u8(<8 x i8> %a, [3 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx3_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx3.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx3.i
}

define <8 x i8> @test_vqtbx4_u8(<8 x i8> %a, [4 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx4_u8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx4.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx4.i
}

define <16 x i8> @test_vqtbx1q_u8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK: test_vqtbx1q_u8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbx1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c)
  ret <16 x i8> %vtbx1.i
}

define <16 x i8> @test_vqtbx2q_u8(<16 x i8> %a, [2 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx2q_u8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx2.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx2.i
}

define <16 x i8> @test_vqtbx3q_u8(<16 x i8> %a, [3 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx3q_u8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx3.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx3.i
}

define <16 x i8> @test_vqtbx4q_u8(<16 x i8> %a, [4 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx4q_u8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx4.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx4.i
}

define <8 x i8> @test_vtbl1_p8(<8 x i8> %a, <8 x i8> %b) {
; CHECK: test_vtbl1_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl11.i
}

define <8 x i8> @test_vqtbl1_p8(<16 x i8> %a, <8 x i8> %b) {
; CHECK: test_vqtbl1_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %a, <8 x i8> %b)
  ret <8 x i8> %vtbl1.i
}

define <8 x i8> @test_vtbl2_p8([2 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl2_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %a.coerce, 1
  %vtbl1.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %b)
  ret <8 x i8> %vtbl17.i
}

define <8 x i8> @test_vqtbl2_p8([2 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl2_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl2.i
}

define <8 x i8> @test_vtbl3_p8([3 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl3_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %a.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %b)
  ret <8 x i8> %vtbl212.i
}

define <8 x i8> @test_vqtbl3_p8([3 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl3_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl3.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl3.i
}

define <8 x i8> @test_vtbl4_p8([4 x <8 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vtbl4_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %a.coerce, 3
  %vtbl2.i = shufflevector <8 x i8> %__a.coerce.fca.0.extract.i, <8 x i8> %__a.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl215.i = shufflevector <8 x i8> %__a.coerce.fca.2.extract.i, <8 x i8> %__a.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl215.i, <8 x i8> %b)
  ret <8 x i8> %vtbl216.i
}

define <8 x i8> @test_vqtbl4_p8([4 x <16 x i8>] %a.coerce, <8 x i8> %b) {
; CHECK: test_vqtbl4_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl4.v8i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <8 x i8> %b)
  ret <8 x i8> %vtbl4.i
}

define <16 x i8> @test_vqtbl1q_p8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: test_vqtbl1q_p8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbl1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl1.v16i8(<16 x i8> %a, <16 x i8> %b)
  ret <16 x i8> %vtbl1.i
}

define <16 x i8> @test_vqtbl2q_p8([2 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl2q_p8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %a.coerce, 1
  %vtbl2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl2.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl2.i
}

define <16 x i8> @test_vqtbl3q_p8([3 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl3q_p8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %a.coerce, 2
  %vtbl3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl3.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl3.i
}

define <16 x i8> @test_vqtbl4q_p8([4 x <16 x i8>] %a.coerce, <16 x i8> %b) {
; CHECK: test_vqtbl4q_p8:
; CHECK: tbl {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__a.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 0
  %__a.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 1
  %__a.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 2
  %__a.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %a.coerce, 3
  %vtbl4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbl4.v16i8(<16 x i8> %__a.coerce.fca.0.extract.i, <16 x i8> %__a.coerce.fca.1.extract.i, <16 x i8> %__a.coerce.fca.2.extract.i, <16 x i8> %__a.coerce.fca.3.extract.i, <16 x i8> %b)
  ret <16 x i8> %vtbl4.i
}

define <8 x i8> @test_vtbx1_p8(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) {
; CHECK: test_vtbx1_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbl1.i = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl11.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl1.v8i8(<16 x i8> %vtbl1.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl11.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx2_p8(<8 x i8> %a, [2 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx2_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <8 x i8>] %b.coerce, 1
  %vtbx1.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx17.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %vtbx1.i, <8 x i8> %c)
  ret <8 x i8> %vtbx17.i
}

define <8 x i8> @test_vtbx3_p8(<8 x i8> %a, [3 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx3_p8:
; CHECK: tbl {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <8 x i8>] %b.coerce, 2
  %vtbl2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl211.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbl212.i = tail call <8 x i8> @llvm.aarch64.neon.vtbl2.v8i8(<16 x i8> %vtbl2.i, <16 x i8> %vtbl211.i, <8 x i8> %c)
  %0 = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
  %1 = sext <8 x i1> %0 to <8 x i8>
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %1, <8 x i8> %a, <8 x i8> %vtbl212.i)
  ret <8 x i8> %vbsl.i
}

define <8 x i8> @test_vtbx4_p8(<8 x i8> %a, [4 x <8 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vtbx4_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <8 x i8>] %b.coerce, 3
  %vtbx2.i = shufflevector <8 x i8> %__b.coerce.fca.0.extract.i, <8 x i8> %__b.coerce.fca.1.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx215.i = shufflevector <8 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %__b.coerce.fca.3.extract.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %vtbx216.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %vtbx2.i, <16 x i8> %vtbx215.i, <8 x i8> %c)
  ret <8 x i8> %vtbx216.i
}

define <8 x i8> @test_vqtbx1_p8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) {
; CHECK: test_vqtbx1_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %vtbx1.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c)
  ret <8 x i8> %vtbx1.i
}

define <8 x i8> @test_vqtbx2_p8(<8 x i8> %a, [2 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx2_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx2.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx2.i
}

define <8 x i8> @test_vqtbx3_p8(<8 x i8> %a, [3 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx3_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx3.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx3.i
}

define <8 x i8> @test_vqtbx4_p8(<8 x i8> %a, [4 x <16 x i8>] %b.coerce, <8 x i8> %c) {
; CHECK: test_vqtbx4_p8:
; CHECK: tbx {{v[0-9]+}}.8b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.8b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <8 x i8> @llvm.aarch64.neon.vtbx4.v8i8(<8 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <8 x i8> %c)
  ret <8 x i8> %vtbx4.i
}

define <16 x i8> @test_vqtbx1q_p8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
; CHECK: test_vqtbx1q_p8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %vtbx1.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c)
  ret <16 x i8> %vtbx1.i
}

define <16 x i8> @test_vqtbx2q_p8(<16 x i8> %a, [2 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx2q_p8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [2 x <16 x i8>] %b.coerce, 1
  %vtbx2.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx2.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx2.i
}

define <16 x i8> @test_vqtbx3q_p8(<16 x i8> %a, [3 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx3q_p8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [3 x <16 x i8>] %b.coerce, 2
  %vtbx3.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx3.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx3.i
}

define <16 x i8> @test_vqtbx4q_p8(<16 x i8> %a, [4 x <16 x i8>] %b.coerce, <16 x i8> %c) {
; CHECK: test_vqtbx4q_p8:
; CHECK: tbx {{v[0-9]+}}.16b, {{{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b}, {{v[0-9]+}}.16b
entry:
  %__b.coerce.fca.0.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 0
  %__b.coerce.fca.1.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 1
  %__b.coerce.fca.2.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 2
  %__b.coerce.fca.3.extract.i = extractvalue [4 x <16 x i8>] %b.coerce, 3
  %vtbx4.i = tail call <16 x i8> @llvm.aarch64.neon.vtbx4.v16i8(<16 x i8> %a, <16 x i8> %__b.coerce.fca.0.extract.i, <16 x i8> %__b.coerce.fca.1.extract.i, <16 x i8> %__b.coerce.fca.2.extract.i, <16 x i8> %__b.coerce.fca.3.extract.i, <16 x i8> %c)
  ret <16 x i8> %vtbx4.i
}

