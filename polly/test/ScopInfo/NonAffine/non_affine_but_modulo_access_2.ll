; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i % 2] = A[i % 2 + 1];
;    }
;
; CHECK: ReadAccess := [Reduction Type: NONE]
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-1 - i0 + o0)/2): 2e0 = -1 - i0 + o0 and o0 >= 1 and o0 <= 2) };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE]
; CHECK:     { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-i0 + o0)/2): 2e0 = -i0 + o0 and o0 >= 0 and o0 <= 1) };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %rem = srem i32 %i.0, 2
  %add = add nsw i32 %rem, 1
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  %tmp = load i32* %arrayidx, align 4
  %rem1 = and i32 %i.0, 1
  %idxprom2 = sext i32 %rem1 to i64
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %idxprom2
  store i32 %tmp, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
