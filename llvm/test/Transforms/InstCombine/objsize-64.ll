; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare noalias i8* @malloc(i32) nounwind
declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readonly

; CHECK: @f1
define i64 @f1() {
  %call = call i8* @malloc(i32 4)
  %size = call i64 @llvm.objectsize.i64(i8* %call, i1 false)
; CHECK-NEXT: ret i64 4
  ret i64 %size
}
