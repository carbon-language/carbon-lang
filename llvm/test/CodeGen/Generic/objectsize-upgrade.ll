; RUN: opt < %s -verify -S | FileCheck %s
; check automatic upgrade of objectsize. To be removed in LLVM 3.3.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define i32 @foo() nounwind {
; CHECK: @foo
  %1 = alloca i8, align 4
  %2 = getelementptr inbounds i8* %1, i32 0
; CHECK: llvm.objectsize.i32(i8* %2, i1 false, i32 0)
  %3 = call i32 @llvm.objectsize.i32(i8* %2, i1 0)
  ret i32 %3
}

; CHECK: @llvm.objectsize.i32(i8*, i1, i32)
declare i32 @llvm.objectsize.i32(i8*, i1) nounwind readonly
