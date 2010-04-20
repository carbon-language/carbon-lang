; RUN: llc < %s -mtriple=i386-apple-darwin
; rdar://7869290

%struct.anon = type { float }

define void @func() nounwind ssp {
entry:
  br label %bb66

bb:                                               ; preds = %bb66
  br i1 undef, label %bb65, label %bb2

bb2:                                              ; preds = %bb
  br i1 undef, label %bb65, label %bb3

bb3:                                              ; preds = %bb2
  br i1 undef, label %bb65, label %bb4

bb4:                                              ; preds = %bb3
  br i1 undef, label %bb65, label %bb5

bb5:                                              ; preds = %bb4
  br i1 undef, label %bb65, label %bb6

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb65, label %bb11

bb11:                                             ; preds = %bb6
  br i1 undef, label %bb65, label %bb12

bb12:                                             ; preds = %bb11
  br i1 undef, label %bb65, label %bb13

bb13:                                             ; preds = %bb12
  br i1 undef, label %bb65, label %bb14

bb14:                                             ; preds = %bb13
  %0 = trunc i16 undef to i1                      ; <i1> [#uses=1]
  %1 = load i8* undef, align 8                    ; <i8> [#uses=1]
  %2 = shl i8 %1, 4                               ; <i8> [#uses=1]
  %3 = lshr i8 %2, 7                              ; <i8> [#uses=1]
  %4 = trunc i8 %3 to i1                          ; <i1> [#uses=1]
  %5 = icmp ne i1 %0, %4                          ; <i1> [#uses=1]
  br i1 %5, label %bb65, label %bb15

bb15:                                             ; preds = %bb14
  %6 = load %struct.anon** undef, align 8         ; <%struct.anon*> [#uses=0]
  br label %bb65

bb65:                                             ; preds = %bb15, %bb14, %bb13, %bb12, %bb11, %bb6, %bb5, %bb4, %bb3, %bb2, %bb
  br label %bb66

bb66:                                             ; preds = %bb65, %entry
  br i1 undef, label %bb, label %bb67

bb67:                                             ; preds = %bb66
  ret void
}
