; RUN: opt < %s -value-propagation -S | FileCheck %s
; PR2581

; CHECK: @run
define i32 @run(i1 %C) nounwind  {
        br i1 %C, label %exit, label %body

body:           ; preds = %0
; CHECK-NOT: select
        %A = select i1 %C, i32 10, i32 11               ; <i32> [#uses=1]
; CHECK: ret i32 11
        ret i32 %A

exit:           ; preds = %0
; CHECK: ret i32 10
        ret i32 10
}