; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze %s | FileCheck %s -check-prefix=INDVAR
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze %s | FileCheck %s

;void f(long a[][128], long N, long M) {
;  long i, j;
;  for (j = 0; j < rnd(); ++j)
;    for (i = 0; i < N; ++i)
;        a[i][j] = 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f([128 x i64]* nocapture %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=1]
  %1 = icmp sgt i64 %0, 0                         ; <i1> [#uses=1]
  br i1 %1, label %bb.nph8, label %return

bb.nph8:                                          ; preds = %entry
  %2 = icmp sgt i64 %N, 0                         ; <i1> [#uses=1]
  br i1 %2, label %bb2.preheader.us, label %bb2.preheader

bb2.preheader.us:                                 ; preds = %bb2.bb3_crit_edge.us, %bb.nph8
  %3 = phi i64 [ 0, %bb.nph8 ], [ %tmp, %bb2.bb3_crit_edge.us ] ; <i64> [#uses=2]
  %tmp = add i64 %3, 1                            ; <i64> [#uses=2]
  br label %bb1.us

bb1.us:                                           ; preds = %bb1.us, %bb2.preheader.us
  %i.06.us = phi i64 [ 0, %bb2.preheader.us ], [ %4, %bb1.us ] ; <i64> [#uses=2]
  %scevgep = getelementptr [128 x i64]* %a, i64 %i.06.us, i64 %3 ; <i64*> [#uses=1]
  store i64 0, i64* %scevgep, align 8
  %4 = add nsw i64 %i.06.us, 1                    ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %4, %N                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2.bb3_crit_edge.us, label %bb1.us

bb2.bb3_crit_edge.us:                             ; preds = %bb1.us
  %5 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=1]
  %6 = icmp sgt i64 %5, %tmp                      ; <i1> [#uses=1]
  br i1 %6, label %bb2.preheader.us, label %return

bb2.preheader:                                    ; preds = %bb2.preheader, %bb.nph8
  %j.07 = phi i64 [ %tmp9, %bb2.preheader ], [ 0, %bb.nph8 ] ; <i64> [#uses=1]
  %tmp9 = add i64 %j.07, 1                        ; <i64> [#uses=2]
  %7 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=1]
  %8 = icmp sgt i64 %7, %tmp9                     ; <i1> [#uses=1]
  br i1 %8, label %bb2.preheader, label %return

return:                                           ; preds = %bb2.preheader, %bb2.bb3_crit_edge.us, %entry
  ret void
}

declare i64 @rnd(...)

; INDVAR: Scop: bb1.us => bb2.bb3_crit_edge.us Parameters: (%N, {0,+,1}<%bb2.preheader.us>, ), Max Loop Depth: 1
; CHECK: Scop: bb1.us => bb2.bb3_crit_edge.us Parameters: (%N, {0,+,1}<%bb2.preheader.us>, ), Max Loop Depth: 1
