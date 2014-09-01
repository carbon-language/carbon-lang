; RUN: opt -basicaa -memcpyopt -instcombine -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo([8 x i64]* noalias nocapture sret dereferenceable(64)) {
entry-block:
  %a = alloca [8 x i64], align 8
  %1 = bitcast [8 x i64]* %a to i8*
  call void @llvm.lifetime.start(i64 64, i8* %1)
  call void @llvm.memset.p0i8.i64(i8* %1, i8 0, i64 64, i32 8, i1 false)
  %2 = bitcast [8 x i64]* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %1, i64 64, i32 8, i1 false)
  call void @llvm.lifetime.end(i64 64, i8* %1)
  ret void

; CHECK-LABEL: @foo(
; CHECK: %1 = bitcast
; CHECK: call void @llvm.memset
; CHECK-NOT: call void @llvm.memcpy
; CHECK: ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
