; RUN: opt -basicaa -loop-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

; CHECK-LABEL: fore_aft_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @fore_aft_less(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, -1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}

; CHECK-LABEL: fore_aft_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @fore_aft_eq(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, 0
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_aft_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @fore_aft_more(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, 1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: fore_sub_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @fore_sub_less(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add7.us2 = add nuw nsw i32 %i, -1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}

; CHECK-LABEL: fore_eq_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @fore_eq_less(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add7.us2 = add nuw nsw i32 %i, 0
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}

; CHECK-LABEL: fore_sub_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @fore_sub_more(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add7.us2 = add nuw nsw i32 %i, 1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %add6.us = add nuw nsw i32 %j, 1
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}

; CHECK-LABEL: sub_aft_less
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @sub_aft_less(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, -1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}

; CHECK-LABEL: sub_aft_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
; CHECK: %j.2 = phi
; CHECK: %j.3 = phi
define void @sub_aft_eq(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, 0
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_aft_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @sub_aft_more(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %add7.us2 = add nuw nsw i32 %i, 1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_less
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @sub_sub_less(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %add7.us2 = add nuw nsw i32 %i, -1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_eq
; CHECK: %j = phi
; CHECK: %j.1 = phi
define void @sub_sub_eq(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %add7.us2 = add nuw nsw i32 %i, 0
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}


; CHECK-LABEL: sub_sub_more
; CHECK: %j = phi
; CHECK-NOT: %j.1 = phi
define void @sub_sub_more(i32* noalias nocapture %A, i32 %N, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp sgt i32 %N, 0
  br i1 %cmp, label %for.outer, label %cleanup

for.outer:
  %i = phi i32 [ %add7.us, %for.latch ], [ 0, %entry ]
  br label %for.inner

for.inner:
  %j = phi i32 [ %add6.us, %for.inner ], [ 0, %for.outer ]
  %sum = phi i32 [ %add.us, %for.inner ], [ 0, %for.outer ]
  %arrayidx5.us = getelementptr inbounds i32, i32* %B, i32 %j
  %0 = load i32, i32* %arrayidx5.us, align 4
  %mul.us = mul nsw i32 %0, %i
  %add.us = add nsw i32 %mul.us, %sum
  %add6.us = add nuw nsw i32 %j, 1
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %i
  store i32 1, i32* %arrayidx.us, align 4
  %add7.us2 = add nuw nsw i32 %i, 1
  %arrayidx8.us = getelementptr inbounds i32, i32* %A, i32 %add7.us2
  store i32 %add.us, i32* %arrayidx8.us, align 4
  %exitcond.us = icmp eq i32 %add6.us, %N
  br i1 %exitcond.us, label %for.latch, label %for.inner

for.latch:
  %add7.us = add nuw nsw i32 %i, 1
  %exitcond29.us = icmp eq i32 %add7.us, %N
  br i1 %exitcond29.us, label %cleanup, label %for.outer

cleanup:
  ret void
}
