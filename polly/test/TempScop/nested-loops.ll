; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* nocapture %a) nounwind {
entry:
  br label %for.i

for.i:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %for.j ]
  %i.inc = add nsw i64 %i, 1
  %exitcond.i = icmp sge i64 %i.inc, 2048
  br i1 %exitcond.i, label %return, label %for.j

for.j:
  %j = phi i64 [ 0, %for.i ], [ %j.inc, %body ]
  %j.inc = add nsw i64 %j, 1
  %exitcond.j = icmp slt i64 %j.inc, 1024
  br i1 %exitcond.j, label %body, label %for.i

body:
  %scevgep = getelementptr i64* %a, i64 %j
  store i64 %j, i64* %scevgep
  br label %for.j

return:
  ret void
}

; CHECK: body
; CHECK:        Domain :=
; CHECK:           { Stmt_body[i0, i1] : i0 >= 0 and i0 <= 2046 and i1 >= 0 and i1 <= 1022 }
