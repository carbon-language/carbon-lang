; RUN: llc < %s -mcpu=x86-64 -x86-experimental-vector-widening-legalization -x86-experimental-vector-shuffle-lowering | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define <4 x i32> @zext_v4i8_to_v4i32(<4 x i8>* %ptr) {
; CHECK-LABEL: zext_v4i8_to_v4i32:
; 
; CHECK:      movd (%{{.*}}), %[[X:xmm[0-9]+]]
; CHECK-NEXT: pxor %[[Z:xmm[0-9]+]], %[[Z]]
; CHECK-NEXT: punpcklbw %[[Z]], %[[X]]
; CHECK-NEXT: punpcklwd %[[Z]], %[[X]]
; CHECK-NEXT: ret

  %val = load <4 x i8>* %ptr
  %ext = zext <4 x i8> %val to <4 x i32>
  ret <4 x i32> %ext
}
