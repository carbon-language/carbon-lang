; RUN: llc -march=hexagon -mcpu=hexagonv4 -O2 < %s | FileCheck %s
; ModuleID = 'hwloop-const.c'
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-linux-gnu"

@b = common global [25000 x i32] zeroinitializer, align 8
@a = common global [25000 x i32] zeroinitializer, align 8
@c = common global [25000 x i32] zeroinitializer, align 8

define i32 @hwloop_bug() nounwind {
entry:
  br label %for.body

; CHECK: endloop
for.body:                                         ; preds = %for.body, %entry
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [25000 x i32]* @b, i32 0, i32 %i.02
  store i32 %i.02, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [25000 x i32]* @a, i32 0, i32 %i.02
  store i32 %i.02, i32* %arrayidx1, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 25000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}
