; RUN: opt %loadPolly %defaultOpts -polly-prepare -polly-analyze-ir  -analyze %s | FileCheck %s

;void f(long a[], long n, long m) {
; long i0, i1;
; for (i0 = 0; i0 < 2 * n + m; ++i0)//loop0
;   a[i0] = n;

; for (i1 = 0; i1 < i0 + m; ++i1)//loop1
;   a[i1] += 2;
;}


; ModuleID = '/tmp/webcompile/_19162_0.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-linux-gnu"

define void @_Z1fPlll(i64* nocapture %a, i64 %n, i64 %m) nounwind {
entry:
  %0 = shl i64 %n, 1                              ; <i64> [#uses=1]
  %1 = add nsw i64 %0, %m                         ; <i64> [#uses=3]
  %2 = icmp sgt i64 %1, 0                         ; <i1> [#uses=1]
  br i1 %2, label %bb, label %bb4.preheader

bb:                                               ; preds = %bb, %entry
  %i0.07 = phi i64 [ %3, %bb ], [ 0, %entry ]     ; <i64> [#uses=2]
  %scevgep11 = getelementptr i64* %a, i64 %i0.07  ; <i64*> [#uses=1]
  store i64 %n, i64* %scevgep11, align 8
  %3 = add nsw i64 %i0.07, 1                      ; <i64> [#uses=2]
  %exitcond10 = icmp eq i64 %3, %1                ; <i1> [#uses=1]
  br i1 %exitcond10, label %bb4.preheader, label %bb

bb4.preheader:                                    ; preds = %bb, %entry
  %i0.0.lcssa = phi i64 [ 0, %entry ], [ %1, %bb ] ; <i64> [#uses=1]
  %4 = add nsw i64 %i0.0.lcssa, %m                ; <i64> [#uses=2]
  %5 = icmp sgt i64 %4, 0                         ; <i1> [#uses=1]
  br i1 %5, label %bb3, label %return

bb3:                                              ; preds = %bb3, %bb4.preheader
  %i1.06 = phi i64 [ %8, %bb3 ], [ 0, %bb4.preheader ] ; <i64> [#uses=2]
  %scevgep = getelementptr i64* %a, i64 %i1.06    ; <i64*> [#uses=2]
  %6 = load i64* %scevgep, align 8                ; <i64> [#uses=1]
  %7 = add nsw i64 %6, 2                          ; <i64> [#uses=1]
  store i64 %7, i64* %scevgep, align 8
  %8 = add nsw i64 %i1.06, 1                      ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %8, %4                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb3

return:                                           ; preds = %bb3, %bb4.preheader
  ret void
}


; CHECK: Scop: entry.split => bb4.preheader.region Parameters: (%m, %n, ), Max Loop
