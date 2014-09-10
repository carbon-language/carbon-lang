; RUN: opt < %s  -loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -debug-only=loop-vectorize -stats -S 2>&1 | FileCheck %s
; REQUIRES: asserts

;
; We have 2 loops, one of them is vectorizable and the second one is not.
;

; CHECK: 2 loop-vectorize               - Number of loops analyzed for vectorization
; CHECK: 1 loop-vectorize               - Number of loops vectorized

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @vectorized(float* nocapture %a, i64 %size) {
entry:
  %cmp1 = icmp sgt i64 %size, 0
  br i1 %cmp1, label %for.header, label %for.end

for.header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %cmp2 = icmp sgt i64 %indvars.iv, %size
  br i1 %cmp2, label %for.end, label %for.body

for.body:

  %arrayidx = getelementptr inbounds float* %a, i64 %indvars.iv
  %0 = load float* %arrayidx, align 4
  %mul = fmul float %0, %0
  store float %mul, float* %arrayidx, align 4

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.header

for.end:
  ret void
}

define void @not_vectorized(float* nocapture %a, i64 %size) {
entry:
  %cmp1 = icmp sgt i64 %size, 0
  br i1 %cmp1, label %for.header, label %for.end

for.header:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %cmp2 = icmp sgt i64 %indvars.iv, %size
  br i1 %cmp2, label %for.end, label %for.body

for.body:

  %0 = add nsw i64 %indvars.iv, -5
  %arrayidx = getelementptr inbounds float* %a, i64 %0
  %1 = load float* %arrayidx, align 4
  %2 = add nsw i64 %indvars.iv, 2
  %arrayidx2 = getelementptr inbounds float* %a, i64 %2
  %3 = load float* %arrayidx2, align 4
  %mul = fmul float %1, %3
  %arrayidx4 = getelementptr inbounds float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx4, align 4

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.header

for.end:
  ret void
}
