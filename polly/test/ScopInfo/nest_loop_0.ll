; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir -analyze %s | FileCheck %s

;void f(long a[][128], long N, long M) {
;  long i, j;
;  for (j = 0; j < M; ++j)
;    for (i = 0; i < N; ++i)
;        ...
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f([128 x i64]* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = icmp sgt i64 %M, 0                         ; <i1> [#uses=1]
  %1 = icmp sgt i64 %N, 0                         ; <i1> [#uses=1]
  %or.cond = and i1 %0, %1                        ; <i1> [#uses=1]
  br i1 %or.cond, label %bb2.preheader, label %return

bb1:                                              ; preds = %bb2.preheader, %bb1
  %i.06 = phi i64 [ 0, %bb2.preheader ], [ %2, %bb1 ] ; <i64> [#uses=3]
  %scevgep = getelementptr [128 x i64]* %a, i64 %i.06, i64 %4 ; <i64*> [#uses=1]
  %tmp = add i64 %i.06, %N                        ; <i64> [#uses=1]
  store i64 %tmp, i64* %scevgep, align 8
  %2 = add nsw i64 %i.06, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %2, %N                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb3, label %bb1

bb3:                                              ; preds = %bb1
  %3 = add i64 %4, 1                              ; <i64> [#uses=2]
  %exitcond9 = icmp eq i64 %3, %M                 ; <i1> [#uses=1]
  br i1 %exitcond9, label %return, label %bb2.preheader

bb2.preheader:                                    ; preds = %bb3, %entry
  %4 = phi i64 [ %3, %bb3 ], [ 0, %entry ]        ; <i64> [#uses=2]
  br label %bb1

return:                                           ; preds = %bb3, %entry
  ret void
}

; CHECK: Scop: bb2.preheader => return.single_exit Parameters: (%M, %N, ), Max Loop Depth: 2
