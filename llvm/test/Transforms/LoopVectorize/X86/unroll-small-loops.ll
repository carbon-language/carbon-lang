; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx -force-vector-width=4 -force-vector-unroll=0 -dce -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"
;CHECK-LABEL: @foo(
;CHECK: load <4 x i32>
;CHECK-NOT: load <4 x i32>
;CHECK: store <4 x i32>
;CHECK-NOT: store <4 x i32>
;CHECK: ret
define i32 @foo(i32* nocapture %A) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = add nsw i32 %3, 6
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 100
  br i1 %exitcond, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 undef
}

;CHECK-LABEL: @bar(
;CHECK: store <4 x i32>
;CHECK: store <4 x i32>
;CHECK: ret
define i32 @bar(i32* nocapture %A, i32 %n) nounwind uwtable ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32* %A, i64 %indvars.iv
  %3 = load i32* %2, align 4
  %4 = add nsw i32 %3, 6
  store i32 %4, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}
