; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze %s | FileCheck %s


;void f(long a[], long N) {
;  long M = rnd();
;  long i;

;  for (i = 0; i < M; ++i)
;   a[i] = i;

;  a[N] = 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* nocapture %a, i64 %N) nounwind {
entry:
  %0 = tail call i64 (...)* @rnd() nounwind       ; <i64> [#uses=2]
  %1 = icmp sgt i64 %0, 0                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %bb2

bb:                                               ; preds = %bb, %entry
  %2 = phi i64 [ 0, %entry ], [ %3, %bb ]         ; <i64> [#uses=3]
  %scevgep = getelementptr i64* %a, i64 %2        ; <i64*> [#uses=1]
  store i64 %2, i64* %scevgep, align 8
  %3 = add nsw i64 %2, 1                          ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %3, %0                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  %4 = getelementptr inbounds i64* %a, i64 %N     ; <i64*> [#uses=1]
  store i64 0, i64* %4, align 8
  ret void
}

declare i64 @rnd(...)

; CHECK: Scop: bb => bb2.single_exit Parameters: (%0, ), Max Loop Depth: 1
