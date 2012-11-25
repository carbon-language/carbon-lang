; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -dce -instcombine -licm -S | FileCheck %s

; int __attribute__((noinline)) sum_array(int *A, int n) {
;  return std::accumulate(A, A + n, 0);
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: @sum_array
;CHECK: phi <4 x i32>
;CHECK: load <4 x i32>
;CHECK: add nsw <4 x i32>
;CHECK: ret i32
define i32 @sum_array(i32* %A, i32 %n) nounwind uwtable readonly noinline ssp {
  %1 = sext i32 %n to i64
  %2 = getelementptr inbounds i32* %A, i64 %1
  %3 = icmp eq i32 %n, 0
  br i1 %3, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

.lr.ph.i:                                         ; preds = %0, %.lr.ph.i
  %.03.i = phi i32* [ %6, %.lr.ph.i ], [ %A, %0 ]
  %.012.i = phi i32 [ %5, %.lr.ph.i ], [ 0, %0 ]
  %4 = load i32* %.03.i, align 4
  %5 = add nsw i32 %4, %.012.i
  %6 = getelementptr inbounds i32* %.03.i, i64 1
  %7 = icmp eq i32* %6, %2
  br i1 %7, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %.lr.ph.i

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %.lr.ph.i, %0
  %.01.lcssa.i = phi i32 [ 0, %0 ], [ %5, %.lr.ph.i ]
  ret i32 %.01.lcssa.i
}
