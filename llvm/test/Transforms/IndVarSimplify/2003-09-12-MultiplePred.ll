; RUN: opt < %s -indvars -S | grep indvar

define i32 @test() {
; <label>:0
        br i1 true, label %LoopHead, label %LoopHead

LoopHead:               ; preds = %LoopHead, %0, %0
        %A = phi i32 [ 7, %0 ], [ 7, %0 ], [ %B, %LoopHead ]            ; <i32> [#uses=1]
        %B = add i32 %A, 1              ; <i32> [#uses=2]
        br i1 true, label %LoopHead, label %Out

Out:            ; preds = %LoopHead
        ret i32 %B
}

