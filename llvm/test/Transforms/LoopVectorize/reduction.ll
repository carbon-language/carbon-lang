; RUN: opt < %s  -loop-vectorize -dce -instcombine -licm -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: @reduction_sum
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add <4 x i32>
;CHECK: ret i32
define i32 @reduction_sum(i32 %n, i32* noalias nocapture %A, i32* noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = getelementptr inbounds i32* %B, i64 %indvars.iv
  %5 = load i32* %4, align 4
  %6 = trunc i64 %indvars.iv to i32
  %7 = add i32 %sum.02, %6
  %8 = add i32 %7, %3
  %9 = add i32 %8, %5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}

;CHECK: @reduction_prod
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: ret i32
define i32 @reduction_prod(i32 %n, i32* noalias nocapture %A, i32* noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %prod.02 = phi i32 [ %9, %.lr.ph ], [ 1, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = getelementptr inbounds i32* %B, i64 %indvars.iv
  %5 = load i32* %4, align 4
  %6 = trunc i64 %indvars.iv to i32
  %7 = mul i32 %prod.02, %6
  %8 = mul i32 %7, %3
  %9 = mul i32 %8, %5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %prod.0.lcssa = phi i32 [ 1, %0 ], [ %9, %.lr.ph ]
  ret i32 %prod.0.lcssa
}

;CHECK: @reduction_mix
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: ret i32
define i32 @reduction_mix(i32 %n, i32* noalias nocapture %A, i32* noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = getelementptr inbounds i32* %B, i64 %indvars.iv
  %5 = load i32* %4, align 4
  %6 = mul nsw i32 %5, %3
  %7 = trunc i64 %indvars.iv to i32
  %8 = add i32 %sum.02, %7
  %9 = add i32 %8, %6
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}

;CHECK: @reduction_bad
;CHECK-NOT: <4 x i32>
;CHECK: ret i32
define i32 @reduction_bad(i32 %n, i32* noalias nocapture %A, i32* noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = getelementptr inbounds i32* %B, i64 %indvars.iv
  %5 = load i32* %4, align 4
  %6 = trunc i64 %indvars.iv to i32
  %7 = add i32 %3, %6
  %8 = add i32 %7, %5
  %9 = mul i32 %8, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}
