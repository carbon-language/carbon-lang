; Various test cases to ensure basic functionality is working for GCSE

; RUN: llvm-as < %s | opt -gcse

define void @testinsts(i32 %i, i32 %j, i32* %p) {
        %A = bitcast i32 %i to i32              ; <i32> [#uses=0]
        %B = bitcast i32 %i to i32              ; <i32> [#uses=0]
        %C = shl i32 %i, 1              ; <i32> [#uses=0]
        %D = shl i32 %i, 1              ; <i32> [#uses=0]
        %E = getelementptr i32* %p, i64 12              ; <i32*> [#uses=0]
        %F = getelementptr i32* %p, i64 12              ; <i32*> [#uses=0]
        %G = getelementptr i32* %p, i64 13              ; <i32*> [#uses=0]
        ret void
}

; Test different combinations of domination properties...
define void @sameBBtest(i32 %i, i32 %j) {
        %A = add i32 %i, %j             ; <i32> [#uses=1]
        %B = add i32 %i, %j             ; <i32> [#uses=1]
        %C = xor i32 %A, -1             ; <i32> [#uses=0]
        %D = xor i32 %B, -1             ; <i32> [#uses=0]
        %E = xor i32 %j, -1             ; <i32> [#uses=0]
        ret void
}

define i32 @dominates(i32 %i, i32 %j) {
        %A = add i32 %i, %j             ; <i32> [#uses=0]
        br label %BB2

BB2:            ; preds = %0
        %B = add i32 %i, %j             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @hascommondominator(i32 %i, i32 %j) {
        br i1 true, label %BB1, label %BB2

BB1:            ; preds = %0
        %A = add i32 %i, %j             ; <i32> [#uses=1]
        ret i32 %A

BB2:            ; preds = %0
        %B = add i32 %i, %j             ; <i32> [#uses=1]
        ret i32 %B
}

