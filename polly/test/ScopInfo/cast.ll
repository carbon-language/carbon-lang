; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze %s | FileCheck %s
;void f(long a[], long N, long M) {
;  long i, j, k;
;  for (j = 0; j < M; ++j)
;    ((long*)j)[(long)a] = j;

;  for (j = 0; j < N; ++j)
;    a[j] = (char)(M + j);
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* %a, i64 %N, i64 %M) nounwind {
entry:
  %0 = icmp sgt i64 %M, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph8, label %bb4.loopexit

bb.nph8:                                          ; preds = %entry
  %1 = ptrtoint i64* %a to i64                    ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph8
  %2 = phi i64 [ 0, %bb.nph8 ], [ %5, %bb ]       ; <i64> [#uses=3]
  %3 = inttoptr i64 %2 to i64*                    ; <i64*> [#uses=1]
  %4 = getelementptr inbounds i64* %3, i64 %1     ; <i64*> [#uses=1]
  store i64 %2, i64* %4, align 8
  %5 = add nsw i64 %2, 1                          ; <i64> [#uses=2]
  %exitcond10 = icmp eq i64 %5, %M                ; <i1> [#uses=1]
  br i1 %exitcond10, label %bb4.loopexit, label %bb

bb3:                                              ; preds = %bb4.loopexit, %bb3
  %j.16 = phi i64 [ 0, %bb4.loopexit ], [ %7, %bb3 ] ; <i64> [#uses=3]
  %scevgep = getelementptr i64* %a, i64 %j.16     ; <i64*> [#uses=1]
  %tmp = add i64 %j.16, %M                        ; <i64> [#uses=1]
  %tmp9 = trunc i64 %tmp to i8                    ; <i8> [#uses=1]
  %6 = sext i8 %tmp9 to i64                       ; <i64> [#uses=1]
  store i64 %6, i64* %scevgep, align 8
  %7 = add nsw i64 %j.16, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %7, %N                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb3

bb4.loopexit:                                     ; preds = %bb, %entry
  %8 = icmp sgt i64 %N, 0                         ; <i1> [#uses=1]
  br i1 %8, label %bb3, label %return

return:                                           ; preds = %bb4.loopexit, %bb3
  ret void
}

; CHECK: Scop: bb4.loopexit => return    Parameters: (%N, ), Max Loop Depth: 1
