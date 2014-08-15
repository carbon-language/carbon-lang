; RUN: opt %loadPolly -polly-dce -polly-ast -analyze < %s | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
; S1:    A[i % 2] = i;
;      for (int i = 0; i < 1024; i++)
; S2:    A[i2] = i;
;    }
;
; CHECK: for (int c1 = 0; c1 <= 1023; c1 += 1)
; CHECK:   Stmt_S2(c1);

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %S1, label %next

S1:
  %rem = srem i32 %i.0, 2
  %arrayidx = getelementptr inbounds i32* %A, i32 %rem
  store i32 %i.0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

next:
 br label %for.cond.2

for.cond.2:
  %i.2 = phi i32 [ 0, %next ], [ %inc.2, %for.inc.2 ]
  %exitcond.2 = icmp ne i32 %i.2, 1024
  br i1 %exitcond.2, label %S2, label %for.end

S2:
  %arrayidx.2 = getelementptr inbounds i32* %A, i32 %i.2
  store i32 %i.2, i32* %arrayidx.2, align 4
  br label %for.inc.2

for.inc.2:
  %inc.2 = add nsw i32 %i.2, 1
  br label %for.cond.2

for.end:
  ret void
}

