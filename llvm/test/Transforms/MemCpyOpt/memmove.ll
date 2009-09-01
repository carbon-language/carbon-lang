; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | FileCheck %s
; These memmoves should get optimized to memcpys.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.0"

declare void @llvm.memmove.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

define i8* @test1(i8* nocapture %src) nounwind {
entry:
; CHECK: @test1
; CHECK: call void @llvm.memcpy

  %call3 = malloc [13 x i8]                       ; <[13 x i8]*> [#uses=1]
  %call3.sub = getelementptr inbounds [13 x i8]* %call3, i64 0, i64 0 ; <i8*> [#uses=2]
  tail call void @llvm.memmove.i64(i8* %call3.sub, i8* %src, i64 13, i32 1)
  ret i8* %call3.sub
}

define void @test2(i8* %P) nounwind {
entry:
; CHECK: @test2
; CHECK: call void @llvm.memcpy
  %add.ptr = getelementptr i8* %P, i64 16         ; <i8*> [#uses=1]
  tail call void @llvm.memmove.i64(i8* %P, i8* %add.ptr, i64 16, i32 1)
  ret void
}

; This cannot be optimize because the src/dst really do overlap.
define void @test3(i8* %P) nounwind {
entry:
; CHECK: @test3
; CHECK: call void @llvm.memmove
  %add.ptr = getelementptr i8* %P, i64 16         ; <i8*> [#uses=1]
  tail call void @llvm.memmove.i64(i8* %P, i8* %add.ptr, i64 17, i32 1)
  ret void
}
