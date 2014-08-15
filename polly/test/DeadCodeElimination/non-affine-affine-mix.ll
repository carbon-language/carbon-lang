; RUN: opt %loadPolly -polly-allow-nonaffine -polly-dce -polly-ast -analyze < %s | FileCheck %s
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
; S1:    A[i ^ 2] = i;
;      for (int i = 0; i < 1024; i++)
; S2:    A[i] = i;
;    }

; We unfortunately do need to execute all iterations of S1, as we do not know
; the size of A and as a result S1 may write for example to A[1024], which
; is not overwritten by S2.

; CHECK: for (int c1 = 0; c1 <= 1023; c1 += 1)
; CHECK:   Stmt_S1(c1);
; CHECK: for (int c1 = 0; c1 <= 1023; c1 += 1)
; CHECK:   Stmt_S2(c1);

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond1 = icmp ne i32 %i.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %S1

S1:                                               ; preds = %for.body
  %xor = xor i32 %i.0, 2
  %idxprom = sext i32 %xor to i64
  %arrayidx = getelementptr inbounds i32* %A, i64 %idxprom
  store i32 %i.0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %S1
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc7, %for.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc7 ], [ 0, %for.end ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body4, label %for.end9

for.body4:                                        ; preds = %for.cond2
  br label %S2

S2:                                               ; preds = %for.body4
  %arrayidx6 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %tmp = trunc i64 %indvars.iv to i32
  store i32 %tmp, i32* %arrayidx6, align 4
  br label %for.inc7

for.inc7:                                         ; preds = %S2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond2

for.end9:                                         ; preds = %for.cond2
  ret void
}
