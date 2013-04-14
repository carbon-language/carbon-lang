; RUN: opt < %s -basicaa -slp-vectorizer -slp-threshold=1000 -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Check that the command line flag works.
;CHECK:rollable
;CHECK-NOT:load <4 x i32>
;CHECK: ret

define i32 @rollable(i32* noalias nocapture %in, i32* noalias nocapture %out, i64 %n) {
  %1 = icmp eq i64 %n, 0
  br i1 %1, label %._crit_edge, label %.lr.ph

.lr.ph:                                           ; preds = %0, %.lr.ph
  %i.019 = phi i64 [ %26, %.lr.ph ], [ 0, %0 ]
  %2 = shl i64 %i.019, 2
  %3 = getelementptr inbounds i32* %in, i64 %2
  %4 = load i32* %3, align 4
  %5 = or i64 %2, 1
  %6 = getelementptr inbounds i32* %in, i64 %5
  %7 = load i32* %6, align 4
  %8 = or i64 %2, 2
  %9 = getelementptr inbounds i32* %in, i64 %8
  %10 = load i32* %9, align 4
  %11 = or i64 %2, 3
  %12 = getelementptr inbounds i32* %in, i64 %11
  %13 = load i32* %12, align 4
  %14 = mul i32 %4, 7
  %15 = add i32 %14, 7
  %16 = mul i32 %7, 7
  %17 = add i32 %16, 14
  %18 = mul i32 %10, 7
  %19 = add i32 %18, 21
  %20 = mul i32 %13, 7
  %21 = add i32 %20, 28
  %22 = getelementptr inbounds i32* %out, i64 %2
  store i32 %15, i32* %22, align 4
  %23 = getelementptr inbounds i32* %out, i64 %5
  store i32 %17, i32* %23, align 4
  %24 = getelementptr inbounds i32* %out, i64 %8
  store i32 %19, i32* %24, align 4
  %25 = getelementptr inbounds i32* %out, i64 %11
  store i32 %21, i32* %25, align 4
  %26 = add i64 %i.019, 1
  %exitcond = icmp eq i64 %26, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret i32 undef
}
