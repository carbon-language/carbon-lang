; RUN: llc < %s -disable-cgp-delete-dead-blocks | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readnone

define void @test5() nounwind optsize noinline ssp {
entry:
; CHECK: movq ___stack_chk_guard@GOTPCREL(%rip)
  %buf = alloca [64 x i8], align 16
  %0 = call i64 @llvm.objectsize.i64(i8* undef, i1 false)
  br i1 false, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  ret void
}
