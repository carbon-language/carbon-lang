; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -dce -instcombine -licm -S -enable-if-conversion | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; When we scalarize a store, make sure that the addresses are extracted
; from a vector. We had a bug where the addresses were the old scalar
; addresses.

; CHECK: @foo
; CHECK: select
; CHECK: extractelement
; CHECK-NEXT: store
; CHECK: extractelement
; CHECK-NEXT: store
; CHECK: extractelement
; CHECK-NEXT: store
; CHECK: extractelement
; CHECK-NEXT: store
; CHECK: ret
define i32 @foo(i32* nocapture %a) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %7, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %7 ]
  %2 = mul nsw i64 %indvars.iv, 7
  %3 = getelementptr inbounds i32* %a, i64 %2
  %4 = load i32* %3, align 4
  %5 = icmp sgt i32 %4, 3
  br i1 %5, label %6, label %7

; <label>:6                                       ; preds = %1
  %tmp = add i32 %4, 4
  %tmp1 = mul i32 %tmp, %4
  br label %7

; <label>:7                                       ; preds = %6, %1
  %x.0 = phi i32 [ %tmp1, %6 ], [ %4, %1 ]
  %8 = add nsw i32 %x.0, 3
  store i32 %8, i32* %3, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %9, label %1

; <label>:9                                       ; preds = %7
  ret i32 0
}
