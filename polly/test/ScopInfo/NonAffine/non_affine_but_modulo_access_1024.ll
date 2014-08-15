; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void jd(int *A, int c) {
;      for (int i = 0; i < 1024; i++)
;        A[i % 1024] = A[(i + 1) % 1024] + A[c % 1024] + A[(i - c) % 1024];
;    }
;
; CHECK: ReadAccess := [Reduction Type: NONE]
; CHECK:     [c] -> { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-1 - i0 + o0)/1024): 1024e0 = -1 - i0 + o0 and o0 >= 0 and o0 <= 1023) };
; CHECK: ReadAccess := [Reduction Type: NONE]
; CHECK:     [c] -> { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-c + o0)/1024): 1024e0 = -c + o0 and o0 >= 0 and o0 <= 1023) };
; CHECK: ReadAccess := [Reduction Type: NONE]
; CHECK:     [c] -> { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-1023c - i0 + o0)/1024): 1024e0 = -1023c - i0 + o0 and o0 >= 0 and o0 <= 1023) };
; CHECK: MustWriteAccess :=  [Reduction Type: NONE]
; CHECK:     [c] -> { Stmt_for_body[i0] -> MemRef_A[o0] : exists (e0 = floor((-i0 + o0)/1024): 1024e0 = -i0 + o0 and o0 >= 0 and o0 <= 1023) };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add nsw i32 %i.0, 1
  %rem = srem i32 %add, 1024
  %idxprom = sext i32 %rem to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  %tmp = load i32* %arrayidx, align 4
  %rem1 = srem i32 %c, 1024
  %idxprom2 = sext i32 %rem1 to i64
  %arrayidx3 = getelementptr inbounds i32* %A, i64 %idxprom2
  %tmp1 = load i32* %arrayidx3, align 4
  %add4 = add nsw i32 %tmp, %tmp1
  %sub = sub nsw i32 %i.0, %c
  %rem5 = and i32 %sub, 1023
  %idxprom6 = sext i32 %rem5 to i64
  %arrayidx7 = getelementptr inbounds i32* %A, i64 %idxprom6
  %tmp2 = load i32* %arrayidx7, align 4
  %add8 = add nsw i32 %add4, %tmp2
  %rem9 = srem i32 %i.0, 1024
  %idxprom10 = sext i32 %rem9 to i64
  %arrayidx11 = getelementptr inbounds i32* %A, i64 %idxprom10
  store i32 %add8, i32* %arrayidx11, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret void
}
