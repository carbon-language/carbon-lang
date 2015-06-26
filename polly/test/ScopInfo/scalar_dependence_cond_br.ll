; RUN: opt %loadPolly -polly-detect-unprofitable -polly-scops -analyze < %s | FileCheck %s
;
;    void f(int *A, int c, int d) {
;      for (int i = 0; i < 1024; i++)
;        if (c < i)
;          A[i]++;
;    }
;
; We should move operands as close to their use as possible, hence in this case
; there should not be any scalar dependence anymore after %cmp1 is moved to 
; %for.body (%c and %indvar.iv are synthesis able).
;
; CHECK-NOT:      [Scalar: 1]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i64 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  %cmp1 = icmp slt i64 %c, %indvars.iv
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
