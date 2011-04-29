; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

define void @f(i32* nocapture %a) nounwind {
entry:
  %0 = tail call i32 (...)* @rnd() nounwind       ; <i32> [#uses=2]
; CHECK: 1 * %0 + 0 * 1
  %1 = icmp sgt i32 %0, 0                         ; <i1> [#uses=1]
  br i1 %1, label %bb, label %return

bb:                                               ; preds = %bb, %entry
  %i.03 = phi i32 [ 0, %entry ], [ %3, %bb ]      ; <i32> [#uses=1]
; CHECK: 1 * {0,+,1}<nuw><nsw><%bb> + 0 * 1
  %2 = tail call i32 (...)* @rnd() nounwind       ; <i32> [#uses=0]
; CHECK: 1 * %2 + 0 * 1
  %3 = add nsw i32 %i.03, 1                       ; <i32> [#uses=2]
; CHECK: 1 * {0,+,1}<nuw><nsw><%bb> + 1 * 1
  %exitcond = icmp eq i32 %3, %0                  ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

declare i32 @rnd(...)
