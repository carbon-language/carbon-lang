; RUN: opt %loadPolly %defaultOpts -polly-prepare -polly-analyze-ir -analyze %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-linux-gnu"

;long f(long a[], long n) {
;  long i, k;
;  k = 0;
;  for (i = 1; i < n; ++i) {
;   k += a[i];
;  }
;  return k;
;}


target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-linux-gnu"

define i64 @f(i64* nocapture %a, i64 %n) nounwind readonly {
entry:
  %0 = icmp sgt i64 %n, 1                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph, label %bb2

bb.nph:                                           ; preds = %entry
  %tmp = add i64 %n, -1                           ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp6, %bb ] ; <i64> [#uses=1]
  %k.05 = phi i64 [ 0, %bb.nph ], [ %2, %bb ]     ; <i64> [#uses=1]
  %tmp6 = add i64 %indvar, 1                      ; <i64> [#uses=3]
  %scevgep = getelementptr i64* %a, i64 %tmp6     ; <i64*> [#uses=1]
  %1 = load i64* %scevgep, align 8                ; <i64> [#uses=1]
  %2 = add nsw i64 %1, %k.05                      ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %tmp6, %tmp             ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb, %entry
  %k.0.lcssa = phi i64 [ 0, %entry ], [ %2, %bb ] ; <i64> [#uses=1]
  ret i64 %k.0.lcssa
}

; CHECK:  Bounds of Loop: bb:   { 1 * %n + -2 }
; CHECK:    BB: bb{
; CHECK:      Reads %k.05.reg2mem[0]  Refs: Must alias {%k.05.reg2mem, } May alias {},
; CHECK:      Reads %a[8 * {0,+,1}<%bb> + 8]  Refs: Must alias {%a, } May alias {},
; CHECK:      Writes %k.0.lcssa.reg2mem[0]  Refs: Must alias {%k.0.lcssa.reg2mem, } May alias {},
; CHECK:      Writes %k.05.reg2mem[0]  Refs: Must alias {%k.05.reg2mem, } May alias {},
; CHECK:    }
