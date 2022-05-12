; REQUIRES: asserts
; RUN: opt < %s -passes='print<delinearization>' -disable-output -debug 2>&1 2>&1 | FileCheck %s
; void foo (int m, int n, char *A) {
;    for (int i=0; i < m; i++)
;      for(int j=0; j< n; j++)
;        A[i*n+j] += 1;
;}

; ModuleID = 'delin.cpp'
;target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
;target triple = "aarch64--linux-gnu"

; CHECK-LABEL: Delinearization on function foo
; CHECK: Inst:  %4 = load i8, i8* %arrayidx.us, align 1
; CHECK: Subscripts
; CHECK-NEXT: {0,+,1}<nuw><nsw><%for.body3.lr.ph.us>
; CHECK-NEXT: {0,+,1}<nuw><nsw><%for.body3.us>
; CHECK: succeeded to delinearize

define void @foo(i32 %m, i32 %n, i8* nocapture %A) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp15 = icmp sgt i32 %m, 0
  %cmp213 = icmp sgt i32 %n, 0
  %or.cond = and i1 %cmp15, %cmp213
  br i1 %or.cond, label %for.cond1.preheader.lr.ph.split.us, label %for.end8

for.cond1.preheader.lr.ph.split.us:               ; preds = %entry.split
  %0 = add i32 %n, -1
  %1 = sext i32 %n to i64
  %2 = add i32 %m, -1
  br label %for.body3.lr.ph.us

for.body3.us:                                     ; preds = %for.body3.us, %for.body3.lr.ph.us
  %indvars.iv = phi i64 [ 0, %for.body3.lr.ph.us ], [ %indvars.iv.next, %for.body3.us ]
  %3 = add nsw i64 %indvars.iv, %5
  %arrayidx.us = getelementptr inbounds i8, i8* %A, i64 %3
  %4 = load i8, i8* %arrayidx.us, align 1
  %add4.us = add i8 %4, 1
  store i8 %add4.us, i8* %arrayidx.us, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.cond1.for.inc6_crit_edge.us, label %for.body3.us

for.body3.lr.ph.us:                               ; preds = %for.cond1.for.inc6_crit_edge.us, %for.cond1.preheader.lr.ph.split.us
  %indvars.iv19 = phi i64 [ %indvars.iv.next20, %for.cond1.for.inc6_crit_edge.us ], [ 0, %for.cond1.preheader.lr.ph.split.us ]
  %5 = mul nsw i64 %indvars.iv19, %1
  br label %for.body3.us

for.cond1.for.inc6_crit_edge.us:                  ; preds = %for.body3.us
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %lftr.wideiv22 = trunc i64 %indvars.iv19 to i32
  %exitcond23 = icmp eq i32 %lftr.wideiv22, %2
  br i1 %exitcond23, label %for.end8.loopexit, label %for.body3.lr.ph.us

for.end8.loopexit:                                ; preds = %for.cond1.for.inc6_crit_edge.us
  br label %for.end8

for.end8:                                         ; preds = %for.end8.loopexit, %entry.split
  ret void
}
