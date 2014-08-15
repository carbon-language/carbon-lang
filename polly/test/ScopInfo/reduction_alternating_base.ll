; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; CHECK: ReadAccess := [Reduction Type: +]
; CHECK:   { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-i0 + o0)/2): 2e0 = -i0 + o0 and o0 <= 1 and o0 >= 0) };
; CHECK: MustWriteAccess :=  [Reduction Type: +]
; CHECK:   { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-i0 + o0)/2): 2e0 = -i0 + o0 and o0 <= 1 and o0 >= 0) };
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i % 2] += i;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %rem = srem i32 %i.0, 2
  %arrayidx = getelementptr inbounds i32* %A, i32 %rem
  %tmp = load i32* %arrayidx, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
