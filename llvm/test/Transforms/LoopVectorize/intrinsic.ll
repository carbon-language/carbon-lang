; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -dce -instcombine -licm -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;CHECK: @sqrt_f32
;CHECK: llvm.sqrt.v4f32
;CHECK: ret void
define void @sqrt_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.sqrt.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone

;CHECK: @sqrt_f64
;CHECK: llvm.sqrt.v4f64
;CHECK: ret void
define void @sqrt_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.sqrt.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.sqrt.f64(double) nounwind readnone

;CHECK: @sin_f32
;CHECK: llvm.sin.v4f32
;CHECK: ret void
define void @sin_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.sin.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sin.f32(float) nounwind readnone

;CHECK: @sin_f64
;CHECK: llvm.sin.v4f64
;CHECK: ret void
define void @sin_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.sin.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.sin.f64(double) nounwind readnone

;CHECK: @cos_f32
;CHECK: llvm.cos.v4f32
;CHECK: ret void
define void @cos_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.cos.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.cos.f32(float) nounwind readnone

;CHECK: @cos_f64
;CHECK: llvm.cos.v4f64
;CHECK: ret void
define void @cos_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.cos.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.cos.f64(double) nounwind readnone

;CHECK: @exp_f32
;CHECK: llvm.exp.v4f32
;CHECK: ret void
define void @exp_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.exp.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.exp.f32(float) nounwind readnone

;CHECK: @exp_f64
;CHECK: llvm.exp.v4f64
;CHECK: ret void
define void @exp_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.exp.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.exp.f64(double) nounwind readnone

;CHECK: @exp2_f32
;CHECK: llvm.exp2.v4f32
;CHECK: ret void
define void @exp2_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.exp2.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.exp2.f32(float) nounwind readnone

;CHECK: @exp2_f64
;CHECK: llvm.exp2.v4f64
;CHECK: ret void
define void @exp2_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.exp2.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.exp2.f64(double) nounwind readnone

;CHECK: @log_f32
;CHECK: llvm.log.v4f32
;CHECK: ret void
define void @log_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.log.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log.f32(float) nounwind readnone

;CHECK: @log_f64
;CHECK: llvm.log.v4f64
;CHECK: ret void
define void @log_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.log.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log.f64(double) nounwind readnone

;CHECK: @log10_f32
;CHECK: llvm.log10.v4f32
;CHECK: ret void
define void @log10_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.log10.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log10.f32(float) nounwind readnone

;CHECK: @log10_f64
;CHECK: llvm.log10.v4f64
;CHECK: ret void
define void @log10_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.log10.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log10.f64(double) nounwind readnone

;CHECK: @log2_f32
;CHECK: llvm.log2.v4f32
;CHECK: ret void
define void @log2_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.log2.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log2.f32(float) nounwind readnone

;CHECK: @log2_f64
;CHECK: llvm.log2.v4f64
;CHECK: ret void
define void @log2_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.log2.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log2.f64(double) nounwind readnone

;CHECK: @fabs_f32
;CHECK: llvm.fabs.v4f32
;CHECK: ret void
define void @fabs_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.fabs.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.fabs.f32(float) nounwind readnone

define void @fabs_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.fabs(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.fabs(double) nounwind readnone

;CHECK: @floor_f32
;CHECK: llvm.floor.v4f32
;CHECK: ret void
define void @floor_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.floor.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.floor.f32(float) nounwind readnone

;CHECK: @floor_f64
;CHECK: llvm.floor.v4f64
;CHECK: ret void
define void @floor_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.floor.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.floor.f64(double) nounwind readnone

;CHECK: @ceil_f32
;CHECK: llvm.ceil.v4f32
;CHECK: ret void
define void @ceil_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.ceil.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.ceil.f32(float) nounwind readnone

;CHECK: @ceil_f64
;CHECK: llvm.ceil.v4f64
;CHECK: ret void
define void @ceil_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.ceil.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.ceil.f64(double) nounwind readnone

;CHECK: @trunc_f32
;CHECK: llvm.trunc.v4f32
;CHECK: ret void
define void @trunc_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.trunc.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.trunc.f32(float) nounwind readnone

;CHECK: @trunc_f64
;CHECK: llvm.trunc.v4f64
;CHECK: ret void
define void @trunc_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.trunc.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.trunc.f64(double) nounwind readnone

;CHECK: @rint_f32
;CHECK: llvm.rint.v4f32
;CHECK: ret void
define void @rint_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.rint.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.rint.f32(float) nounwind readnone

;CHECK: @rint_f64
;CHECK: llvm.rint.v4f64
;CHECK: ret void
define void @rint_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.rint.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.rint.f64(double) nounwind readnone

;CHECK: @nearbyint_f32
;CHECK: llvm.nearbyint.v4f32
;CHECK: ret void
define void @nearbyint_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %call = tail call float @llvm.nearbyint.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.nearbyint.f32(float) nounwind readnone

;CHECK: @nearbyint_f64
;CHECK: llvm.nearbyint.v4f64
;CHECK: ret void
define void @nearbyint_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %call = tail call double @llvm.nearbyint.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.nearbyint.f64(double) nounwind readnone

;CHECK: @fma_f32
;CHECK: llvm.fma.v4f32
;CHECK: ret void
define void @fma_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z, float* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds float* %w, i64 %indvars.iv
  %1 = load float* %arrayidx2, align 4, !tbaa !0
  %arrayidx4 = getelementptr inbounds float* %z, i64 %indvars.iv
  %2 = load float* %arrayidx4, align 4, !tbaa !0
  %3 = tail call float @llvm.fma.f32(float %0, float %2, float %1)
  %arrayidx6 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %3, float* %arrayidx6, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone

;CHECK: @fma_f64
;CHECK: llvm.fma.v4f64
;CHECK: ret void
define void @fma_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z, double* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %arrayidx2 = getelementptr inbounds double* %w, i64 %indvars.iv
  %1 = load double* %arrayidx2, align 8, !tbaa !3
  %arrayidx4 = getelementptr inbounds double* %z, i64 %indvars.iv
  %2 = load double* %arrayidx4, align 8, !tbaa !3
  %3 = tail call double @llvm.fma.f64(double %0, double %2, double %1)
  %arrayidx6 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %3, double* %arrayidx6, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.fma.f64(double, double, double) nounwind readnone

;CHECK: @pow_f32
;CHECK: llvm.pow.v4f32
;CHECK: ret void
define void @pow_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float* %y, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds float* %z, i64 %indvars.iv
  %1 = load float* %arrayidx2, align 4, !tbaa !0
  %call = tail call float @llvm.pow.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.pow.f32(float, float) nounwind readnone

;CHECK: @pow_f64
;CHECK: llvm.pow.v4f64
;CHECK: ret void
define void @pow_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double* %y, i64 %indvars.iv
  %0 = load double* %arrayidx, align 8, !tbaa !3
  %arrayidx2 = getelementptr inbounds double* %z, i64 %indvars.iv
  %1 = load double* %arrayidx2, align 8, !tbaa !3
  %call = tail call double @llvm.pow.f64(double %0, double %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx4, align 8, !tbaa !3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.pow.f64(double, double) nounwind readnone

!0 = metadata !{metadata !"float", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
!3 = metadata !{metadata !"double", metadata !1}
!4 = metadata !{metadata !"int", metadata !1}
