; RUN: opt < %s -loop-reduce -S | FileCheck %s

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  br i1 undef, label %bb4.preheader, label %bb.nph8

bb4.preheader:                                    ; preds = %entry
  br label %bb4

bb1:                                              ; preds = %bb4
  br i1 undef, label %bb.nph8, label %bb3

bb3:                                              ; preds = %bb1
  %phitmp = add i32 %indvar, 1                    ; <i32> [#uses=1]
  br label %bb4

bb4:                                              ; preds = %bb3, %bb4.preheader
; CHECK: %lsr.iv = phi
; CHECK: %lsr.iv.next = add i32 %lsr.iv, 1
; CHECK: %0 = icmp slt i32 %lsr.iv.next, %argc
  %indvar = phi i32 [ 1, %bb4.preheader ], [ %phitmp, %bb3 ] ; <i32> [#uses=2]
  %0 = icmp slt i32 %indvar, %argc                ; <i1> [#uses=1]
  br i1 %0, label %bb1, label %bb.nph8

bb.nph8:                                          ; preds = %bb4, %bb1, %entry
  unreachable
}
