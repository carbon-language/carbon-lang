; Test kernel hwasan instrumentation for alloca.
;
; RUN: opt < %s -passes=hwasan -hwasan-kernel=1 -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use32(i32*)

define void @test_alloca() sanitize_hwaddress {
; CHECK-LABEL: @test_alloca(
; CHECK: %[[FP:[^ ]*]] = call i8* @llvm.frameaddress.p0i8(i32 0)
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %[[FP]] to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 20
; CHECK: %[[BASE_TAG:[^ ]*]] = xor i64 %[[A]], %[[B]]

; CHECK: %[[X:[^ ]*]] = alloca { i32, [12 x i8] }, align 16
; CHECK: %[[X_BC:[^ ]*]] = bitcast { i32, [12 x i8] }* %[[X]] to i32*
; CHECK: %[[X_TAG:[^ ]*]] = xor i64 %[[BASE_TAG]], 0
; CHECK: %[[X1:[^ ]*]] = ptrtoint i32* %[[X_BC]] to i64
; CHECK: %[[C:[^ ]*]] = shl i64 %[[X_TAG]], 56
; CHECK: %[[D:[^ ]*]] = or i64 %[[C]], 72057594037927935
; CHECK: %[[E:[^ ]*]] = and i64 %[[X1]], %[[D]]
; CHECK: %[[X_HWASAN:[^ ]*]] = inttoptr i64 %[[E]] to i32*

entry:
  %x = alloca i32, align 4
  call void @use32(i32* nonnull %x)
  ret void
}
