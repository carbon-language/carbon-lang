; RUN: opt -S -basicaa -loop-idiom < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

declare i64 @foo() nounwind

; Nested loops
define void @test1(i8* nocapture %A, i64 %n) nounwind {
entry:
  %call8 = tail call i64 @foo() nounwind
  %tobool9 = icmp eq i64 %call8, 0
  br i1 %tobool9, label %while.end, label %for.cond.preheader.lr.ph

for.cond.preheader.lr.ph:                         ; preds = %entry
  %cmp6 = icmp eq i64 %n, 0
  br label %for.cond.preheader

while.cond.loopexit:                              ; preds = %for.body, %for.cond.preheader
  %call = tail call i64 @foo() nounwind
  %tobool = icmp eq i64 %call, 0
  br i1 %tobool, label %while.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %for.cond.preheader.lr.ph, %while.cond.loopexit
  br i1 %cmp6, label %while.cond.loopexit, label %for.body

for.body:                                         ; preds = %for.cond.preheader, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %for.cond.preheader ]
  %add = add i64 %i.07, 10
  %arrayidx = getelementptr inbounds i8* %A, i64 %add
  %0 = load i8* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8* %A, i64 %i.07
  store i8 %0, i8* %arrayidx1, align 1
  %inc = add i64 %i.07, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %while.cond.loopexit, label %for.body

while.end:                                        ; preds = %while.cond.loopexit, %entry
  ret void

; CHECK: @test1
; CHECK: call void @llvm.memmove.p0i8.p0i8.i64(
}
