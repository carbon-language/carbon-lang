; RUN: opt %loadPolly -basicaa -polly-vectorizer=polly -enable-polly-openmp -polly-opt-isl -polly-codegen < %s

; void f(int *A, int a, int b) {
;   int local = a > b ? a : b;
;   int i;
;   for (i = 0; i < 100; i++) {
;     A[i] += local;
;   }
; }
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32* %A, i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %a, %cond.true ], [ %b, %cond.false ]
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %cond.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %cond.end ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  %tmp = load i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, %cond
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
