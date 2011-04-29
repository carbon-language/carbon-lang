; RUN: opt %loadPolly %defaultOpts -print-scev-affine  -analyze  < %s | FileCheck %s

define void @f([8 x i32]* nocapture %x) nounwind {
entry:
  br label %bb5.preheader

bb2:                                              ; preds = %bb3.preheader, %bb2
  %k.09 = phi i64 [ 0, %bb3.preheader ], [ %1, %bb2 ] ; <i64> [#uses=2]
  %tmp19 = add i64 %k.09, %tmp18                  ; <i64> [#uses=1]
  %scevgep = getelementptr [8 x i32]* %x, i64 2, i64 %tmp19 ; <i32*> [#uses=1]
; CHECK: sizeof(i32) * {0,+,1}<nuw><nsw><%bb2> + (20 * sizeof(i32)) * {0,+,1}<%bb3.preheader> + (35 * sizeof(i32)) * {0,+,1}<%bb5.preheader> + 1 * %x + (18 * sizeof(i32)) * 1
  %0 = tail call i32 (...)* @rnd() nounwind       ; <i32> [#uses=1]
  store i32 %0, i32* %scevgep, align 4
  %1 = add nsw i64 %k.09, 1                       ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %1, 64                  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb4, label %bb2

bb4:                                              ; preds = %bb2
  %2 = add i64 %j.010, 1                          ; <i64> [#uses=2]
  %exitcond20 = icmp eq i64 %2, 64                ; <i1> [#uses=1]
  br i1 %exitcond20, label %bb6, label %bb3.preheader

bb3.preheader:                                    ; preds = %bb5.preheader, %bb4
  %j.010 = phi i64 [ 0, %bb5.preheader ], [ %2, %bb4 ] ; <i64> [#uses=2]
  %tmp21 = mul i64 %j.010, 20                     ; <i64> [#uses=1]
  %tmp18 = add i64 %tmp21, %tmp23                 ; <i64> [#uses=1]
  br label %bb2

bb6:                                              ; preds = %bb4
  %3 = add i64 %i.012, 1                          ; <i64> [#uses=2]
  %exitcond25 = icmp eq i64 %3, 64                ; <i1> [#uses=1]
  br i1 %exitcond25, label %return, label %bb5.preheader

bb5.preheader:                                    ; preds = %bb6, %entry
  %i.012 = phi i64 [ 0, %entry ], [ %3, %bb6 ]    ; <i64> [#uses=2]
  %tmp = mul i64 %i.012, 35                       ; <i64> [#uses=1]
  %tmp23 = add i64 %tmp, 2                        ; <i64> [#uses=1]
  br label %bb3.preheader

return:                                           ; preds = %bb6
  ret void
}

declare i32 @rnd(...)
