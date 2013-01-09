; RUN: opt < %s  -loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -S | FileCheck %s
; RUN: opt < %s  -loop-vectorize -mtriple=thumbv7-apple-ios3.0.0 -mcpu=swift -S | FileCheck %s --check-prefix=SWIFT

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

;CHECK: @foo
;CHECK: load <4 x i32>
;CHECK-NOT: load <4 x i32>
;CHECK: ret
;SWIFT: @foo
;SWIFT: load <4 x i32>
;SWIFT: load <4 x i32>
;SWIFT: ret
define i32 @foo(i32* nocapture %A, i32 %n) nounwind readonly ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %i.02 = phi i32 [ %5, %.lr.ph ], [ 0, %0 ]
  %sum.01 = phi i32 [ %4, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32* %A, i32 %i.02
  %3 = load i32* %2, align 4
  %4 = add nsw i32 %3, %sum.01
  %5 = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %5, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %4, %.lr.ph ]
  ret i32 %sum.0.lcssa
}
