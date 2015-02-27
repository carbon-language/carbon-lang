; RUN: opt %loadPolly -polly-detect-unprofitable -polly-scops -polly-model-phi-nodes -disable-polly-intra-scop-scalar-to-array -analyze < %s | FileCheck %s
;
;    int jd(int *restrict A, int x, int N) {
;      for (int i = 1; i < N; i++)
;        for (int j = 3; j < N; j++)
;          x += A[i];
;      return x;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @jd(i32* noalias %A, i32 %x, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
; CHECK: Stmt_for_cond
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_cond[i0] -> MemRef_x_addr_0[] };
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc4 ], [ 1, %entry ]
  %x.addr.0 = phi i32 [ %x, %entry ], [ %x.addr.1.lcssa, %for.inc4 ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
; CHECK: Stmt_for_body
; CHECK:       ReadAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_body[i0] -> MemRef_x_addr_0[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_body[i0] -> MemRef_x_addr_1[] };
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
; CHECK: Stmt_for_cond1
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_cond1[i0, i1] -> MemRef_x_addr_1_lcssa[] };
  %x.addr.1 = phi i32 [ %x.addr.0, %for.body ], [ %add, %for.inc ]
  %j.0 = phi i32 [ 3, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, %N
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  br label %for.inc

for.inc:                                          ; preds = %for.body3
; CHECK: Stmt_for_inc
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_1[] };
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_inc[i0, i1] -> MemRef_x_addr_1[] };
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK:           [N] -> { Stmt_for_inc[i0, i1] -> MemRef_A[1 + i0] };
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %x.addr.1, %tmp1
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
; CHECK: Stmt_for_end
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_end[i0] -> MemRef_x_addr_1_lcssa[] };
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_end[i0] -> MemRef_x_addr_1_lcssa[] };
  %x.addr.1.lcssa = phi i32 [ %x.addr.1, %for.cond1 ]
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
; CHECK: Stmt_for_inc4
; CHECK:       ReadAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_inc4[i0] -> MemRef_x_addr_1_lcssa[] };
; CHECK:       MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK:           [N] -> { Stmt_for_inc4[i0] -> MemRef_x_addr_0[] };
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret i32 %x.addr.0
}
