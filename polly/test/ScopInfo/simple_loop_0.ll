; RUN: opt %loadPolly %defaultOpts  -polly-analyze-ir  -analyze %s | FileCheck %s -check-prefix=WITHAF
; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir  -analyze %s | FileCheck %s

;void f(long a[], long N) {
;  long i;
;  for (i = 0; i < 128; ++i)
;    a[i] = a[i] - a[i + 2];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* nocapture %a, i64 %N) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %i.03 = phi i64 [ 0, %entry ], [ %3, %bb ]      ; <i64> [#uses=3]
  %scevgep = getelementptr i64* %a, i64 %i.03     ; <i64*> [#uses=2]
  %tmp = add i64 %i.03, 2                        ; <i64> [#uses=1]
  %scevgep4 = getelementptr i64* %a, i64 %tmp     ; <i64*> [#uses=1]
  %0 = load i64* %scevgep, align 8                ; <i64> [#uses=1]
  %1 = load i64* %scevgep4, align 8               ; <i64> [#uses=1]
  %2 = sub i64 %0, %1                             ; <i64> [#uses=1]
  store i64 %2, i64* %scevgep, align 8
  %3 = add nsw i64 %i.03, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %3, 128                 ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb
  ret void
}

; CHECK: Scop: bb => return        Parameters: ()
; WITHAF: Scop: bb => return      Parameters: (), Max Loop Depth: 1
; WITHAF: Bounds of Loop: bb:   { 127 }
; WITHAF:   BB: bb{
; WITHAF:     Reads %a[8 * {0,+,1}<nuw><nsw><%bb> + 0]
; WITHAF:     Reads %a[8 * {0,+,1}<nuw><nsw><%bb> + 16]
; WITHAF:     Writes %a[8 * {0,+,1}<nuw><nsw><%bb> + 0]
; WITHAF:   }
