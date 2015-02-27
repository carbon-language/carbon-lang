; RUN: opt -S -basicaa -licm < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test represents the following function:
; void test1(int * __restrict__ a, int * __restrict__ b, int &c, int n) {
;   for (int i = 0; i < n; ++i)
;     if (a[i] > 0)
;       a[i] = c*b[i];
; }
; and we want to hoist the load of %c out of the loop. This can be done only
; because the dereferenceable attribute is on %c.

; CHECK-LABEL: @test1
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

define void @test1(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly nonnull dereferenceable(4) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This is the same as @test1, but without the dereferenceable attribute on %c.
; Without this attribute, we should not hoist the load of %c.

; CHECK-LABEL: @test2
; CHECK: if.then:
; CHECK: load i32, i32* %c, align 4

define void @test2(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly nonnull %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This test represents the following function:
; void test3(int * restrict a, int * restrict b, int c[static 3], int n) {
;   for (int i = 0; i < n; ++i)
;     if (a[i] > 0)
;       a[i] = c[2]*b[i];
; }
; and we want to hoist the load of c[2] out of the loop. This can be done only
; because the dereferenceable attribute is on %c.

; CHECK-LABEL: @test3
; CHECK: load i32, i32* %c2, align 4
; CHECK: for.body:

define void @test3(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly dereferenceable(12) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c2 = getelementptr inbounds i32, i32* %c, i64 2
  %1 = load i32, i32* %c2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This is the same as @test3, but with a dereferenceable attribute on %c with a
; size too small to cover c[2] (and so we should not hoist it).

; CHECK-LABEL: @test4
; CHECK: if.then:
; CHECK: load i32, i32* %c2, align 4

define void @test4(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly dereferenceable(11) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c2 = getelementptr inbounds i32, i32* %c, i64 2
  %1 = load i32, i32* %c2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { nounwind uwtable }

