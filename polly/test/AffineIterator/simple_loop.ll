; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d, i32* nocapture %x) nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %bb ] ; <i64> [#uses=3]
; CHECK: 1 * {0,+,1}<%bb> + 0 * 1
  %scevgep = getelementptr i32* %x, i64 %indvar   ; <i32*> [#uses=1]
; CHECK: 4 *  {0,+,1}<%bb> + 1 * %x + 0 * 1
  %i.04 = trunc i64 %indvar to i32                ; <i32> [#uses=1]
; CHECK: 1 *  {0,+,1}<%bb> + 0 * 1
  store i32 %i.04, i32* %scevgep, align 4
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
; CHECK: 1 *  {0,+,1}<%bb> + 1 * 1
  %exitcond = icmp eq i64 %indvar.next, 64        ; <i1> [#uses=1]
  br i1 %exitcond, label %bb2, label %bb

bb2:                                              ; preds = %bb
  ret i32 %a
}
