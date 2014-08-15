; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        if (i % 4)
;          A[i] += 1;
;    }
;
; CHECK: Function: jd_and
; CHECK:   Stmt_if_then
; CHECK:     Domain :=
; CHECK:         { Stmt_if_then[i0] : exists (e0 = floor((i0)/4): i0 >= 0 and i0 <= 1023 and 4e0 <= -1 + i0 and 4e0 >= -3 + i0) };
; CHECK:     Scattering :=
; CHECK:         { Stmt_if_then[i0] -> scattering[0, i0, 0] };
; CHECK:     ReadAccess := [Reduction Type: +]
; CHECK:         { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK:     MustWriteAccess :=  [Reduction Type: +]
; CHECK:         { Stmt_if_then[i0] -> MemRef_A[i0] };
;
; CHECK: Function: jd_srem
; CHECK:   Stmt_if_then
; CHECK:     Domain :=
; CHECK:         { Stmt_if_then[i0] : exists (e0 = floor((i0)/4): i0 >= 0 and i0 <= 1023 and 4e0 <= -1 + i0 and 4e0 >= -3 + i0) };
; CHECK:     Scattering :=
; CHECK:         { Stmt_if_then[i0] -> scattering[0, i0, 0] };
; CHECK:     ReadAccess := [Reduction Type: +]
; CHECK:         { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK:     MustWriteAccess :=  [Reduction Type: +]
; CHECK:         { Stmt_if_then[i0] -> MemRef_A[i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd_and(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = trunc i64 %indvars.iv to i32
  %rem1 = and i32 %tmp, 3
  %tobool = icmp eq i32 %rem1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  %tmp2 = load i32* %arrayidx, align 4
  %add = add nsw i32 %tmp2, 1
  store i32 %add, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @jd_srem(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %rem = srem i64 %indvars.iv, 4
  %tobool = icmp eq i64 %rem, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  %tmp2 = load i32* %arrayidx, align 4
  %add = add nsw i32 %tmp2, 1
  store i32 %add, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
