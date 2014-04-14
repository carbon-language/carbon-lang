; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <4 x i32> @copyTuple.QPair(i8* %a, i8* %b) {
; CHECK-LABEL: copyTuple.QPair:
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld2 {{{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8* %a, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> <i32 2, i32 2, i32 2, i32 2>, i32 0, i32 4)
  %extract = extractvalue { <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8* %b, <4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 1, i32 4)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

define <4 x i32> @copyTuple.QTriple(i8* %a, i8* %b, <4 x i32> %c) {
; CHECK-LABEL: copyTuple.QTriple:
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld3 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8* %a, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, i32 0, i32 4)
  %extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8* %b, <4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, i32 1, i32 4)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

define <4 x i32> @copyTuple.QQuad(i8* %a, i8* %b, <4 x i32> %c) {
; CHECK-LABEL: copyTuple.QQuad:
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: orr v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ld4 {{{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s, {{v[0-9]+}}.s}[{{[0-9]+}}], [x{{[0-9]+|sp}}]
entry:
  %vld = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8* %a, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, <4 x i32> %c, i32 0, i32 4)
  %extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld, 0
  %vld1 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8* %b, <4 x i32> %extract, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %c, <4 x i32> %c, i32 1, i32 4)
  %vld1.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } %vld1, 0
  ret <4 x i32> %vld1.fca.0.extract
}

declare { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2lane.v4i32(i8*, <4 x i32>, <4 x i32>, i32, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
declare { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4lane.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32, i32)
