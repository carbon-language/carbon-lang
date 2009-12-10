; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s

define i32 @do_loop(i32* nocapture %sdp, i32* nocapture %ddp, i8* %mdp, i8* nocapture %cdp, i32 %w) nounwind readonly optsize ssp {
entry:
  br label %bb

bb:                                               ; preds = %bb5, %entry
  %mask.1.in = load i8* undef, align 1            ; <i8> [#uses=3]
  %0 = icmp eq i8 %mask.1.in, 0                   ; <i1> [#uses=1]
  br i1 %0, label %bb5, label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
; CHECK: %bb2
; CHECK: movb %ch, %al
  %1 = zext i8 %mask.1.in to i32                  ; <i32> [#uses=1]
  %2 = zext i8 undef to i32                       ; <i32> [#uses=1]
  %3 = mul i32 %2, %1                             ; <i32> [#uses=1]
  %4 = add i32 %3, 1                              ; <i32> [#uses=1]
  %5 = add i32 %4, 0                              ; <i32> [#uses=1]
  %6 = lshr i32 %5, 8                             ; <i32> [#uses=1]
  %retval12.i = trunc i32 %6 to i8                ; <i8> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %mask.0.in = phi i8 [ %retval12.i, %bb2 ], [ %mask.1.in, %bb1 ] ; <i8> [#uses=1]
  %7 = icmp eq i8 %mask.0.in, 0                   ; <i1> [#uses=1]
  br i1 %7, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb5

bb5:                                              ; preds = %bb4, %bb3, %bb
  br i1 undef, label %bb6, label %bb

bb6:                                              ; preds = %bb5
  ret i32 undef
}
