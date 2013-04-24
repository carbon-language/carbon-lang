; RUN: opt -S -loop-vectorize -dce -instcombine -force-vector-width=2 -force-vector-unroll=1  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@A = common global [1024 x i32] zeroinitializer, align 16

; Signed tests.

; Turn this into a max reduction.
; CHECK: @max_red
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>

define i32 @max_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp sgt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a max reduction. The select has its inputs reversed therefore
; this is a max reduction.
; CHECK: @max_red_inverse_select
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>

define i32 @max_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp slt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction.
; CHECK: @min_red
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>

define i32 @min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp slt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction. The select has its inputs reversed therefore
; this is a min reduction.
; CHECK: @min_red_inverse_select
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>

define i32 @min_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp sgt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Unsigned tests.

; Turn this into a max reduction.
; CHECK: @umax_red
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>

define i32 @umax_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp ugt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a max reduction. The select has its inputs reversed therefore
; this is a max reduction.
; CHECK: @umax_red_inverse_select
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>

define i32 @umax_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp ult i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction.
; CHECK: @umin_red
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>

define i32 @umin_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp ult i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; Turn this into a min reduction. The select has its inputs reversed therefore
; this is a min reduction.
; CHECK: @umin_red_inverse_select
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>

define i32 @umin_red_inverse_select(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp ugt i32 %max.red.08, %0
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; SGE -> SLT
; Turn this into a min reduction (select inputs are reversed).
; CHECK: @sge_min_red
; CHECK: icmp sge <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp slt <2 x i32>
; CHECK: select <2 x i1>

define i32 @sge_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp sge i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; SLE -> SGT
; Turn this into a max reduction (select inputs are reversed).
; CHECK: @sle_min_red
; CHECK: icmp sle <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp sgt <2 x i32>
; CHECK: select <2 x i1>

define i32 @sle_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp sle i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; UGE -> ULT
; Turn this into a min reduction (select inputs are reversed).
; CHECK: @uge_min_red
; CHECK: icmp uge <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ult <2 x i32>
; CHECK: select <2 x i1>

define i32 @uge_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp uge i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; ULE -> UGT
; Turn this into a max reduction (select inputs are reversed).
; CHECK: @ule_min_red
; CHECK: icmp ule <2 x i32>
; CHECK: select <2 x i1>
; CHECK: middle.block
; CHECK: icmp ugt <2 x i32>
; CHECK: select <2 x i1>

define i32 @ule_min_red(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %cmp3 = icmp ule i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %max.red.08, i32 %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; No reduction.
; CHECK: @no_red_1
; CHECK-NOT: icmp <2 x i32>
define i32 @no_red_1(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %arrayidx1 = getelementptr inbounds [1024 x i32]* @A, i64 1, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %1 = load i32* %arrayidx1, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %max.red.08
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}

; CHECK: @no_red_2
; CHECK-NOT: icmp <2 x i32>
define i32 @no_red_2(i32 %max) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.red.08 = phi i32 [ %max, %entry ], [ %max.red.0, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32]* @A, i64 0, i64 %indvars.iv
  %arrayidx1 = getelementptr inbounds [1024 x i32]* @A, i64 1, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %1 = load i32* %arrayidx1, align 4
  %cmp3 = icmp sgt i32 %0, %max.red.08
  %max.red.0 = select i1 %cmp3, i32 %0, i32 %1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %max.red.0
}
