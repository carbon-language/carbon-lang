; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @shl.v8i8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: shl.v8i8:
; CHECK: ushl v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %c = shl <8 x i8> %a, %b
  ret <8 x i8> %c
}

define <4 x i16> @shl.v4i16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: shl.v4i16:
; CHECK: ushl v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %c = shl <4 x i16> %a, %b
  ret <4 x i16> %c
}

define <2 x i32> @shl.v2i32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: shl.v2i32:
; CHECK: ushl v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %c = shl <2 x i32> %a, %b
  ret <2 x i32> %c
}

define <1 x i64> @shl.v1i64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: shl.v1i64:
; CHECK: ushl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %c = shl <1 x i64> %a, %b
  ret <1 x i64> %c
}

define <16 x i8> @shl.v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: shl.v16i8:
; CHECK: ushl v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %c = shl <16 x i8> %a, %b
  ret <16 x i8> %c
}

define <8 x i16> @shl.v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: shl.v8i16:
; CHECK: ushl v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %c = shl <8 x i16> %a, %b
  ret <8 x i16> %c
}

define <4 x i32> @shl.v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: shl.v4i32:
; CHECK: ushl v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %c = shl <4 x i32> %a, %b
  ret <4 x i32> %c
}

define <2 x i64> @shl.v2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: shl.v2i64:
; CHECK: ushl v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %c = shl <2 x i64> %a, %b
  ret <2 x i64> %c
}

define <8 x i8> @lshr.v8i8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: lshr.v8i8:
; CHECK: neg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
; CHECK: ushl v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %c = lshr <8 x i8> %a, %b
  ret <8 x i8> %c
}

define <4 x i16> @lshr.v4i16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: lshr.v4i16:
; CHECK: neg v{{[0-9]+}}.4h, v{{[0-9]+}}.4
; CHECK: ushl v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %c = lshr <4 x i16> %a, %b
  ret <4 x i16> %c
}

define <2 x i32> @lshr.v2i32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: lshr.v2i32:
; CHECK: neg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; CHECK: ushl v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %c = lshr <2 x i32> %a, %b
  ret <2 x i32> %c
}

define <1 x i64> @lshr.v1i64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: lshr.v1i64:
; CHECK: neg d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: ushl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %c = lshr <1 x i64> %a, %b
  ret <1 x i64> %c
}

define <16 x i8> @lshr.v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: lshr.v16i8:
; CHECK: neg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: ushl v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %c = lshr <16 x i8> %a, %b
  ret <16 x i8> %c
}

define <8 x i16> @lshr.v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: lshr.v8i16:
; CHECK: neg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; CHECK: ushl v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %c = lshr <8 x i16> %a, %b
  ret <8 x i16> %c
}

define <4 x i32> @lshr.v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: lshr.v4i32:
; CHECK: neg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; CHECK: ushl v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %c = lshr <4 x i32> %a, %b
  ret <4 x i32> %c
}

define <2 x i64> @lshr.v2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: lshr.v2i64:
; CHECK: neg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; CHECK: ushl v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %c = lshr <2 x i64> %a, %b
  ret <2 x i64> %c
}

define <8 x i8> @ashr.v8i8(<8 x i8> %a, <8 x i8> %b) {
; CHECK-LABEL: ashr.v8i8:
; CHECK: neg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
; CHECK: sshl v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %c = ashr <8 x i8> %a, %b
  ret <8 x i8> %c
}

define <4 x i16> @ashr.v4i16(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: ashr.v4i16:
; CHECK: neg v{{[0-9]+}}.4h, v{{[0-9]+}}.4
; CHECK: sshl v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %c = ashr <4 x i16> %a, %b
  ret <4 x i16> %c
}

define <2 x i32> @ashr.v2i32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: ashr.v2i32:
; CHECK: neg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
; CHECK: sshl v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %c = ashr <2 x i32> %a, %b
  ret <2 x i32> %c
}

define <1 x i64> @ashr.v1i64(<1 x i64> %a, <1 x i64> %b) {
; CHECK-LABEL: ashr.v1i64:
; CHECK: neg d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: sshl d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %c = ashr <1 x i64> %a, %b
  ret <1 x i64> %c
}

define <16 x i8> @ashr.v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: ashr.v16i8:
; CHECK: neg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
; CHECK: sshl v{{[0-9]+}}.16b, v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %c = ashr <16 x i8> %a, %b
  ret <16 x i8> %c
}

define <8 x i16> @ashr.v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: ashr.v8i16:
; CHECK: neg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
; CHECK: sshl v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %c = ashr <8 x i16> %a, %b
  ret <8 x i16> %c
}

define <4 x i32> @ashr.v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: ashr.v4i32:
; CHECK: neg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
; CHECK: sshl v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %c = ashr <4 x i32> %a, %b
  ret <4 x i32> %c
}

define <2 x i64> @ashr.v2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: ashr.v2i64:
; CHECK: neg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
; CHECK: sshl v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %c = ashr <2 x i64> %a, %b
  ret <2 x i64> %c
}