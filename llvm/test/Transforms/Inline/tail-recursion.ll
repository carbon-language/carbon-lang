; RUN: opt -inline -tailcallelim -indvars -loop-deletion -S < %s | FileCheck %s

; Inline shouldn't inline foo into itself because it's a tailcallelim
; candidate. Tailcallelim should convert the call into a loop. Indvars
; should calculate the exit value, making the loop dead. Loop deletion
; should delete the loop.
; PR6842

;      CHECK: define i32 @bar() nounwind {
; CHECK-NEXT:     ret i32 10000
; CHECK-NEXT: }

define internal i32 @foo(i32 %x) nounwind {
  %i = add i32 %x, 1                              ; <i32> [#uses=3]
  %a = icmp slt i32 %i, 10000                     ; <i1> [#uses=1]
  br i1 %a, label %more, label %done

done:                                             ; preds = %0
  ret i32 %i

more:                                             ; preds = %0
  %z = tail call i32 @foo(i32 %i)                  ; <i32> [#uses=1]
  ret i32 %z
}

define i32 @bar() nounwind {
  %z = call i32 @foo(i32 0)                        ; <i32> [#uses=1]
  ret i32 %z
}
