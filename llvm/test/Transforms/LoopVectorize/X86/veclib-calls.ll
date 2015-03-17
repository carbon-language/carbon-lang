; RUN: opt < %s -vector-library=Accelerate -loop-vectorize -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;CHECK-LABEL: @sqrt_f32(
;CHECK: vsqrtf{{.*}}<4 x float>
;CHECK: ret void
declare float @sqrtf(float) nounwind readnone
define void @sqrt_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @sqrtf(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;CHECK-LABEL: @exp_f32(
;CHECK: vexpf{{.*}}<4 x float>
;CHECK: ret void
declare float @expf(float) nounwind readnone
define void @exp_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @expf(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;CHECK-LABEL: @log_f32(
;CHECK: vlogf{{.*}}<4 x float>
;CHECK: ret void
declare float @logf(float) nounwind readnone
define void @log_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @logf(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; For abs instruction we'll generate vector intrinsic, as it's cheaper than a lib call.
;CHECK-LABEL: @fabs_f32(
;CHECK: fabs{{.*}}<4 x float>
;CHECK: ret void
declare float @fabsf(float) nounwind readnone
define void @fabs_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @fabsf(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Test that we can vectorize an intrinsic into a vector call.
;CHECK-LABEL: @exp_f32_intrin(
;CHECK: vexpf{{.*}}<4 x float>
;CHECK: ret void
declare float @llvm.exp.f32(float) nounwind readnone
define void @exp_f32_intrin(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.exp.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Test that we don't vectorize arbitrary functions.
;CHECK-LABEL: @foo_f32(
;CHECK-NOT: foo{{.*}}<4 x float>
;CHECK: ret void
declare float @foo(float) nounwind readnone
define void @foo_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @foo(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Test that we don't vectorize calls with nobuiltin attribute.
;CHECK-LABEL: @sqrt_f32_nobuiltin(
;CHECK-NOT: vsqrtf{{.*}}<4 x float>
;CHECK: ret void
define void @sqrt_f32_nobuiltin(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @sqrtf(float %0) nounwind readnone nobuiltin
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
