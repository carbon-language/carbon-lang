; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s
; rdar://7354379

declare arm_apcscc double @floor(double) nounwind readnone

define void @t(i1 %a, double %b) {
entry:
  br i1 %a, label %bb3, label %bb1

bb1:                                              ; preds = %entry
  unreachable

bb3:                                              ; preds = %entry
  br i1 %a, label %bb7, label %bb5

bb5:                                              ; preds = %bb3
  unreachable

bb7:                                              ; preds = %bb3
  br i1 %a, label %bb11, label %bb9

bb9:                                              ; preds = %bb7
; CHECK: @ BB#3:
; CHECK: cbnz
  %0 = tail call arm_apcscc  double @floor(double %b) nounwind readnone ; <double> [#uses=0]
  br label %bb11

bb11:                                             ; preds = %bb9, %bb7
  %1 = getelementptr i32* undef, i32 0
  store i32 0, i32* %1
  ret void
}
