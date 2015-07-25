; RUN: opt < %s -loop-unswitch -loop-unswitch-threshold=0 -verify-loop-info -S < %s 2>&1 | FileCheck %s

; This test contains two trivial unswitch condition in one loop. 
; LoopUnswitch pass should be able to unswitch the second one 
; after unswitching the first one.


; CHECK:  br i1 %cond1, label %..split_crit_edge, label %.loop_exit.split_crit_edge

; CHECK:  ..split_crit_edge:                                ; preds = %0
; CHECK:    br label %.split

; CHECK:  .split:                                           ; preds = %..split_crit_edge
; CHECK:    br i1 %cond2, label %.split..split.split_crit_edge, label %.split.loop_exit.split1_crit_edge

; CHECK:  .split..split.split_crit_edge:                    ; preds = %.split
; CHECK:    br label %.split.split

; CHECK:  .split.split:                                     ; preds = %.split..split.split_crit_edge
; CHECK:    br label %loop_begin

; CHECK:  loop_begin:                                       ; preds = %do_something, %.split.split
; CHECK:    br i1 true, label %continue, label %loop_exit

; CHECK:  continue:                                         ; preds = %loop_begin
; CHECK:    %var_val = load i32, i32* %var
; CHECK:    br i1 true, label %do_something, label %loop_exit

define i32 @test(i32* %var, i1 %cond1, i1 %cond2) {
  br label %loop_begin

loop_begin:  
  br i1 %cond1, label %continue, label %loop_exit	; first trivial condition

continue:
  %var_val = load i32, i32* %var
  br i1 %cond2, label %do_something, label %loop_exit	; second trivial condition  

do_something:
  call void @some_func() noreturn nounwind
  br label %loop_begin

loop_exit:
  ret i32 0
}

declare void @some_func() noreturn