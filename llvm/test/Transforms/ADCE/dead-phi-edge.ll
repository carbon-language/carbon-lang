; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep call

; The call is not live just because the PHI uses the call retval!

define i32 @test(i32 %X) {
; <label>:0
        br label %Done

DeadBlock:              ; No predecessors!
        %Y = call i32 @test( i32 0 )            ; <i32> [#uses=1]
        br label %Done

Done:           ; preds = %DeadBlock, %0
        %Z = phi i32 [ %X, %0 ], [ %Y, %DeadBlock ]             ; <i32> [#uses=1]
        ret i32 %Z
}

