; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void bar();
;    void foo(int *A, int *B, long int N, long int M) {
;      for (long int j = 0; j < M; ++j) {
;        bar();
;        for (long int i = 0; i < N; ++i)
;          A[i] += 1;
;        for (long int i = 0; i < N; ++i)
;          A[i] += 1;
;      }
;    }
;
; Test to check that the scop only counts loop depth for loops fully contained
; in the scop.
; CHECK: Max Loop Depth: 1
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32* %A, i32* %B, i64 %N, i64 %M) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc13, %entry
  %j.0 = phi i64 [ 0, %entry ], [ %inc14, %for.inc13 ]
  %cmp = icmp slt i64 %j.0, %M
  br i1 %cmp, label %for.body, label %for.end15

for.body:                                         ; preds = %for.cond
  call void (...) @bar() #2
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %i.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i64 %i.0, %N
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc10, %for.end
  %i4.0 = phi i64 [ 0, %for.end ], [ %inc11, %for.inc10 ]
  %cmp6 = icmp slt i64 %i4.0, %N
  br i1 %cmp6, label %for.body7, label %for.end12

for.body7:                                        ; preds = %for.cond5
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %i4.0
  %tmp1 = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %tmp1, 1
  store i32 %add9, i32* %arrayidx8, align 4
  br label %for.inc10

for.inc10:                                        ; preds = %for.body7
  %inc11 = add nuw nsw i64 %i4.0, 1
  br label %for.cond5

for.end12:                                        ; preds = %for.cond5
  br label %for.inc13

for.inc13:                                        ; preds = %for.end12
  %inc14 = add nuw nsw i64 %j.0, 1
  br label %for.cond

for.end15:                                        ; preds = %for.cond
  ret void
}

declare void @bar(...) #1

