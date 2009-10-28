; RUN: opt < %s -gvn -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i8 @test(i8* %P) nounwind {
; CHECK: @test
; CHECK-NOT: load
; CHECK: ret i8
entry:
  store i8 1, i8* %P
  %0 = call {}* @llvm.invariant.start(i64 32, i8* %P)
  %1 = tail call i32 @foo(i8* %P)
  call void @llvm.invariant.end({}* %0, i64 32, i8* %P)
  %2 = load i8* %P
  ret i8 %2
}

declare i32 @foo(i8*) nounwind 
declare {}* @llvm.invariant.start(i64 %S, i8* nocapture %P) readonly
declare void @llvm.invariant.end({}* %S, i64 %SS, i8* nocapture %P)